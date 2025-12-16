"""
Cloud HAL Implementation

HAL for serverless environments: AWS Lambda, Google Cloud Run, Azure Functions.
Uses cloud storage (S3, GCS, Blob), cloud APIs for sensors, HTTP for actuators.
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from urllib import request, error

from .base import (
    HAL,
    HALConfig,
    Platform,
    SensorType,
    ActuatorType,
    SensorReading,
)


logger = logging.getLogger(__name__)


class CloudHAL(HAL):
    """
    Cloud HAL for serverless environments.

    Features:
    - S3/GCS/Blob storage for persistence
    - HTTP-based sensors (REST APIs)
    - HTTP-based actuators (webhooks, API calls)
    - Environment variable configuration
    - Stateless operation

    Environment Variables:
    - WOOEDGE_STORAGE_BACKEND: "s3", "gcs", "azure", "memory" (default: "memory")
    - WOOEDGE_S3_BUCKET: S3 bucket name
    - WOOEDGE_GCS_BUCKET: GCS bucket name
    - AWS_REGION: AWS region for S3

    Example (AWS Lambda):
        def handler(event, context):
            hal = CloudHAL()
            hal.initialize()

            # Register API sensor
            hal.register_sensor("weather", SensorType.TEMPERATURE, {
                "type": "http",
                "url": "https://api.weather.com/v1/temp",
                "method": "GET",
                "extract": "$.temperature"
            })

            reading = hal.read_sensor("weather")
            return {"temperature": reading.value}
    """

    def __init__(self, config: HALConfig = None):
        super().__init__(config)
        self._start_time = time.time()
        self._storage_backend: str = "memory"
        self._memory_storage: Dict[str, Any] = {}
        self._s3_client: Optional[Any] = None
        self._gcs_client: Optional[Any] = None

    @property
    def platform(self) -> Platform:
        return Platform.CLOUD

    def initialize(self) -> None:
        """Initialize Cloud HAL."""
        if self._initialized:
            return

        # Detect storage backend
        self._storage_backend = os.environ.get("WOOEDGE_STORAGE_BACKEND", "memory")

        if self._storage_backend == "s3":
            self._init_s3()
        elif self._storage_backend == "gcs":
            self._init_gcs()

        self._initialized = True
        logger.info(f"CloudHAL initialized (storage: {self._storage_backend})")
        self.emit("initialized")

    def _init_s3(self) -> None:
        """Initialize AWS S3 client."""
        try:
            import boto3
            self._s3_client = boto3.client('s3')
            self._s3_bucket = os.environ.get("WOOEDGE_S3_BUCKET", "wooedge-data")
        except ImportError:
            logger.warning("boto3 not available, falling back to memory storage")
            self._storage_backend = "memory"

    def _init_gcs(self) -> None:
        """Initialize Google Cloud Storage client."""
        try:
            from google.cloud import storage
            self._gcs_client = storage.Client()
            self._gcs_bucket = os.environ.get("WOOEDGE_GCS_BUCKET", "wooedge-data")
        except ImportError:
            logger.warning("google-cloud-storage not available, falling back to memory storage")
            self._storage_backend = "memory"

    def shutdown(self) -> None:
        """Shutdown Cloud HAL."""
        self._initialized = False
        self.emit("shutdown")

    # ==================== Time ====================

    def get_time(self) -> float:
        """Get current Unix timestamp."""
        return time.time()

    def get_ticks_ms(self) -> int:
        """Get milliseconds since HAL start."""
        return int((time.time() - self._start_time) * 1000)

    def sleep(self, seconds: float) -> None:
        """Sleep (not recommended in serverless)."""
        time.sleep(seconds)

    def sleep_ms(self, ms: int) -> None:
        """Sleep in milliseconds."""
        time.sleep(ms / 1000.0)

    # ==================== Sensors ====================

    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read sensor value (HTTP API call)."""
        if sensor_id not in self._sensors:
            return None

        sensor_info = self._sensors[sensor_id]
        sensor_type = sensor_info["type"]
        config = sensor_info["config"]

        sensor_hw_type = config.get("type", "").lower()

        if sensor_hw_type == "http":
            value = self._read_http_sensor(config)
        elif sensor_hw_type == "env":
            value = self._read_env_sensor(config)
        elif sensor_hw_type == "mock":
            value = config.get("value")
        else:
            return None

        if value is None:
            return None

        return SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            value=value,
            unit=config.get("unit", ""),
        )

    def _read_http_sensor(self, config: Dict[str, Any]) -> Optional[Any]:
        """Read sensor value from HTTP API."""
        url = config.get("url")
        if not url:
            return None

        method = config.get("method", "GET")
        headers = config.get("headers", {})
        body = config.get("body")
        extract_path = config.get("extract")

        try:
            result = self.http_request(method, url, headers, body, timeout=5.0)
            if result is None:
                return None

            # Extract value from response
            response_body = result.get("body", "")
            try:
                data = json.loads(response_body)
            except:
                return response_body

            if extract_path:
                # Simple JSON path extraction ($.field.subfield)
                value = self._extract_json_path(data, extract_path)
                return value

            return data

        except Exception as e:
            logger.error(f"HTTP sensor error: {e}")
            return None

    def _extract_json_path(self, data: Any, path: str) -> Any:
        """Extract value from JSON using simple path notation."""
        # Remove leading $. if present
        if path.startswith("$."):
            path = path[2:]
        elif path.startswith("$"):
            path = path[1:]

        parts = path.split(".")
        current = data

        for part in parts:
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except:
                    return None
            else:
                return None

        return current

    def _read_env_sensor(self, config: Dict[str, Any]) -> Optional[Any]:
        """Read sensor value from environment variable."""
        var_name = config.get("env_var")
        if not var_name:
            return None

        value = os.environ.get(var_name)
        if value is None:
            return config.get("default")

        # Type conversion
        value_type = config.get("value_type", "string")
        try:
            if value_type == "int":
                return int(value)
            elif value_type == "float":
                return float(value)
            elif value_type == "bool":
                return value.lower() in ("true", "1", "yes")
        except:
            pass

        return value

    # ==================== Actuators ====================

    def execute(
        self,
        actuator_id: str,
        command: str,
        **kwargs,
    ) -> bool:
        """Execute actuator command (HTTP API call)."""
        if actuator_id not in self._actuators:
            return False

        actuator_info = self._actuators[actuator_id]
        config = actuator_info["config"]

        actuator_hw_type = config.get("type", "").lower()

        try:
            if actuator_hw_type == "http" or actuator_hw_type == "webhook":
                return self._execute_http_actuator(config, command, kwargs)
            elif actuator_hw_type == "sns":
                return self._execute_sns_actuator(config, command, kwargs)
            elif actuator_hw_type == "sqs":
                return self._execute_sqs_actuator(config, command, kwargs)
            elif actuator_hw_type == "pubsub":
                return self._execute_pubsub_actuator(config, command, kwargs)
            elif actuator_hw_type == "mock":
                logger.info(f"Mock actuator: {actuator_id}.{command}({kwargs})")
                return True
            else:
                logger.warning(f"Unknown actuator type: {actuator_hw_type}")
                return False

        except Exception as e:
            logger.error(f"Actuator execute error: {e}")
            return False

    def _execute_http_actuator(
        self,
        config: Dict[str, Any],
        command: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Execute HTTP actuator (webhook, API call)."""
        url = config.get("url")
        if not url:
            return False

        method = config.get("method", "POST")
        headers = config.get("headers", {})

        # Build request body
        body = {
            "command": command,
            **kwargs,
        }

        result = self.http_request(method, url, headers, body, timeout=5.0)
        return result is not None and result.get("status", 0) < 400

    def _execute_sns_actuator(
        self,
        config: Dict[str, Any],
        command: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Publish to AWS SNS topic."""
        try:
            import boto3
            sns = boto3.client('sns')
            topic_arn = config.get("topic_arn")

            message = json.dumps({"command": command, **kwargs})
            sns.publish(TopicArn=topic_arn, Message=message)
            return True
        except Exception as e:
            logger.error(f"SNS publish error: {e}")
            return False

    def _execute_sqs_actuator(
        self,
        config: Dict[str, Any],
        command: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Send message to AWS SQS queue."""
        try:
            import boto3
            sqs = boto3.client('sqs')
            queue_url = config.get("queue_url")

            message = json.dumps({"command": command, **kwargs})
            sqs.send_message(QueueUrl=queue_url, MessageBody=message)
            return True
        except Exception as e:
            logger.error(f"SQS send error: {e}")
            return False

    def _execute_pubsub_actuator(
        self,
        config: Dict[str, Any],
        command: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Publish to Google Pub/Sub topic."""
        try:
            from google.cloud import pubsub_v1
            publisher = pubsub_v1.PublisherClient()
            topic_path = config.get("topic_path")

            message = json.dumps({"command": command, **kwargs}).encode('utf-8')
            publisher.publish(topic_path, message)
            return True
        except Exception as e:
            logger.error(f"Pub/Sub publish error: {e}")
            return False

    # ==================== Storage ====================

    def store(self, key: str, value: Any) -> bool:
        """Store value to cloud storage."""
        if self._storage_backend == "memory":
            self._memory_storage[key] = value
            return True
        elif self._storage_backend == "s3":
            return self._store_s3(key, value)
        elif self._storage_backend == "gcs":
            return self._store_gcs(key, value)
        return False

    def _store_s3(self, key: str, value: Any) -> bool:
        """Store to S3."""
        try:
            body = json.dumps(value)
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=f"wooedge/{key}.json",
                Body=body,
                ContentType="application/json",
            )
            return True
        except Exception as e:
            logger.error(f"S3 store error: {e}")
            return False

    def _store_gcs(self, key: str, value: Any) -> bool:
        """Store to GCS."""
        try:
            bucket = self._gcs_client.bucket(self._gcs_bucket)
            blob = bucket.blob(f"wooedge/{key}.json")
            blob.upload_from_string(json.dumps(value), content_type="application/json")
            return True
        except Exception as e:
            logger.error(f"GCS store error: {e}")
            return False

    def load(self, key: str, default: Any = None) -> Any:
        """Load value from cloud storage."""
        if self._storage_backend == "memory":
            return self._memory_storage.get(key, default)
        elif self._storage_backend == "s3":
            return self._load_s3(key, default)
        elif self._storage_backend == "gcs":
            return self._load_gcs(key, default)
        return default

    def _load_s3(self, key: str, default: Any) -> Any:
        """Load from S3."""
        try:
            response = self._s3_client.get_object(
                Bucket=self._s3_bucket,
                Key=f"wooedge/{key}.json",
            )
            body = response['Body'].read().decode('utf-8')
            return json.loads(body)
        except:
            return default

    def _load_gcs(self, key: str, default: Any) -> Any:
        """Load from GCS."""
        try:
            bucket = self._gcs_client.bucket(self._gcs_bucket)
            blob = bucket.blob(f"wooedge/{key}.json")
            return json.loads(blob.download_as_text())
        except:
            return default

    def delete(self, key: str) -> bool:
        """Delete from cloud storage."""
        if self._storage_backend == "memory":
            self._memory_storage.pop(key, None)
            return True
        elif self._storage_backend == "s3":
            try:
                self._s3_client.delete_object(
                    Bucket=self._s3_bucket,
                    Key=f"wooedge/{key}.json",
                )
                return True
            except:
                return False
        elif self._storage_backend == "gcs":
            try:
                bucket = self._gcs_client.bucket(self._gcs_bucket)
                blob = bucket.blob(f"wooedge/{key}.json")
                blob.delete()
                return True
            except:
                return False
        return False

    def list_keys(self, prefix: str = "") -> List[str]:
        """List storage keys."""
        if self._storage_backend == "memory":
            return [k for k in self._memory_storage.keys() if k.startswith(prefix)]
        elif self._storage_backend == "s3":
            try:
                response = self._s3_client.list_objects_v2(
                    Bucket=self._s3_bucket,
                    Prefix=f"wooedge/{prefix}",
                )
                keys = []
                for obj in response.get("Contents", []):
                    key = obj["Key"].replace("wooedge/", "").replace(".json", "")
                    keys.append(key)
                return keys
            except:
                return []
        return []

    # ==================== Networking ====================

    def is_connected(self) -> bool:
        """Cloud is always connected (or not running)."""
        return True

    def http_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        body: Any = None,
        timeout: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request."""
        try:
            headers = headers or {}
            headers.setdefault("User-Agent", "WooEdge-Cloud/1.0")

            data = None
            if body is not None:
                if isinstance(body, dict):
                    data = json.dumps(body).encode('utf-8')
                    headers.setdefault("Content-Type", "application/json")
                elif isinstance(body, str):
                    data = body.encode('utf-8')
                else:
                    data = body

            req = request.Request(url, data=data, headers=headers, method=method)

            with request.urlopen(req, timeout=timeout) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": response.read().decode('utf-8'),
                }

        except error.HTTPError as e:
            return {
                "status": e.code,
                "headers": dict(e.headers),
                "body": e.read().decode('utf-8'),
            }
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return None

    # ==================== Utilities ====================

    def get_free_memory(self) -> int:
        """Get free memory (not meaningful in serverless)."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except:
            return 0

    def get_cpu_freq(self) -> int:
        """Get CPU frequency (not meaningful in serverless)."""
        return 0

    def get_lambda_context(self) -> Dict[str, Any]:
        """Get AWS Lambda context from environment."""
        return {
            "function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME"),
            "function_version": os.environ.get("AWS_LAMBDA_FUNCTION_VERSION"),
            "memory_limit": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
            "log_group": os.environ.get("AWS_LAMBDA_LOG_GROUP_NAME"),
            "log_stream": os.environ.get("AWS_LAMBDA_LOG_STREAM_NAME"),
            "region": os.environ.get("AWS_REGION"),
        }

    def get_cloud_run_context(self) -> Dict[str, Any]:
        """Get Google Cloud Run context from environment."""
        return {
            "service": os.environ.get("K_SERVICE"),
            "revision": os.environ.get("K_REVISION"),
            "configuration": os.environ.get("K_CONFIGURATION"),
            "port": os.environ.get("PORT"),
        }
