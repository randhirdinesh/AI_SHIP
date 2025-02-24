import os
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import timm
from ultralytics import YOLO
import logging
from datetime import datetime
import mlflow
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import threading
from queue import Queue
import time
from typing import Dict, List, Tuple, Optional
import onnx
import onnxruntime

class BiofoulingDataset(Dataset):
    """Custom dataset with advanced augmentation for limited data scenarios"""
    def __init__(self, 
                 data_dir: str, 
                 transform: Optional[A.Compose] = None,
                 is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Load and verify data
        self.images, self.labels = self._load_data()
        self.mixup_alpha = 0.2 if is_training else 0.0
        
        # Initialize cache for faster loading
        self.cache = {}
        self.cache_lock = threading.Lock()

    def _load_data(self) -> Tuple[List[Path], List[Path]]:
        """Load and validate image-label pairs"""
        images_dir = self.data_dir / 'images'
        labels_dir = self.data_dir / 'labels'
        
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        label_paths = []
        
        for img_path in image_paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                logging.warning(f"Missing label for {img_path}")
                continue
            label_paths.append(label_path)
            
        return image_paths, label_paths

    def __len__(self) -> int:
        return len(self.images)

    def _load_and_cache_image(self, idx: int) -> np.ndarray:
        """Load image with caching for efficiency"""
        with self.cache_lock:
            if idx in self.cache:
                return self.cache[idx].copy()
            
            image = cv2.imread(str(self.images[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Cache with size limit
            if len(self.cache) < 100:  # Adjust based on available memory
                self.cache[idx] = image.copy()
                
            return image

    def _parse_label(self, label_path: Path) -> torch.Tensor:
        """Parse YOLO format labels"""
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, *bbox = map(float, line.strip().split())
                labels.append([class_id] + bbox)
        return torch.tensor(labels)

    def _mixup(self, img1: np.ndarray, img2: np.ndarray, 
               label1: torch.Tensor, label2: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Apply mixup augmentation"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img, (lam * label1 + (1 - lam) * label2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_and_cache_image(idx)
        label = self._parse_label(self.labels[idx])
        
        # Apply mixup during training
        if self.is_training and random.random() < 0.5:
            mix_idx = random.randint(0, len(self) - 1)
            mix_image = self._load_and_cache_image(mix_idx)
            mix_label = self._parse_label(self.labels[mix_idx])
            image, label = self._mixup(image, mix_image, label, mix_label)
        
        # Apply augmentation
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return torch.from_numpy(image).permute(2, 0, 1), label

class ProductionConfig:
    """Enhanced configuration for production environment"""
    def __init__(self):
        self.IMG_SIZE = (640, 640)
        self.BATCH_SIZE = 1  # Optimized for real-time inference
        self.CONFIDENCE_THRESHOLD = 0.6
        self.NMS_THRESHOLD = 0.5
        self.MAX_DETECTIONS = 100
        
        # Real-time processing
        self.PROCESS_QUEUE_SIZE = 8
        self.MAX_LATENCY_MS = 33  # Target 30 FPS
        self.WARM_UP_ITERATIONS = 10
        
        # Model optimization
        self.QUANTIZATION = True
        self.TRT_ENABLED = True
        self.MIXED_PRECISION = True
        
        # Hardware specific
        self.NUM_THREADS = 4
        self.CUDA_VISIBLE_DEVICES = "0"
        self.POWER_MODE = "max-n"  # Jetson specific
        
        # Paths
        self.MODEL_PATH = Path("models/production")
        self.CACHE_DIR = Path("cache")
        self.LOG_DIR = Path("logs")
        
        # Initialize directories
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        """Create necessary directories"""
        for path in [self.MODEL_PATH, self.CACHE_DIR, self.LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOG_DIR / f'production_{datetime.now():%Y%m%d}.log'),
                logging.StreamHandler()
            ]
        )

class ProductionModel:
    """Production-optimized model for real-time inference"""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self._initialize_models()
        self._setup_inference_queue()
        
        # Warm up models
        self._warmup()

    def _initialize_models(self):
        """Initialize and optimize models for production"""
        try:
            # Load base models
            self.yolo = YOLO('yolov8s.pt')
            self.efficient = self._load_optimized_model('efficientnetv2_rw_s')
            self.convnext = self._load_optimized_model('convnext_small')
            
            # Optimize for inference
            if self.config.QUANTIZATION:
                self._quantize_models()
            
            if self.config.TRT_ENABLED and torch.cuda.is_available():
                self._convert_to_tensorrt()
                
            self.logger.info("Models initialized and optimized for production")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _load_optimized_model(self, model_name: str) -> torch.nn.Module:
        """Load and optimize a model for inference"""
        model = timm.create_model(model_name, pretrained=False, num_classes=7)
        model.load_state_dict(torch.load(f"models/{model_name}.pth"))
        model = model.to(self.device).eval()
        
        # Optimize model
        if self.device.type == 'cuda':
            model = torch.jit.script(model)
        return model

    def _quantize_models(self):
        """Apply quantization to models"""
        if self.device.type == 'cpu':
            # CPU quantization
            self.efficient = torch.quantization.quantize_dynamic(
                self.efficient, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.convnext = torch.quantization.quantize_dynamic(
                self.convnext, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            # GPU quantization
            self.efficient = torch.quantization.quantize_dynamic(
                self.efficient, {torch.nn.Linear}, dtype=torch.float16
            )
            self.convnext = torch.quantization.quantize_dynamic(
                self.convnext, {torch.nn.Linear}, dtype=torch.float16
            )

    def _convert_to_tensorrt(self):
        """Convert models to TensorRT format"""
        try:
            from torch2trt import torch2trt
            
            # Convert models
            x = torch.ones((1, 3, *self.config.IMG_SIZE)).cuda()
            self.efficient = torch2trt(self.efficient, [x])
            self.convnext = torch2trt(self.convnext, [x])
            
            self.logger.info("Models converted to TensorRT")
        except ImportError:
            self.logger.warning("TensorRT conversion failed - torch2trt not available")

    def _setup_inference_queue(self):
        """Setup queue for parallel inference"""
        self.inference_queue = Queue(maxsize=self.config.PROCESS_QUEUE_SIZE)
        self.result_queue = Queue()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

    def _inference_worker(self):
        """Worker thread for continuous inference"""
        while True:
            try:
                image = self.inference_queue.get()
                if image is None:
                    break
                    
                result = self._process_single_image(image)
                self.result_queue.put(result)
            except Exception as e:
                self.logger.error(f"Inference error: {str(e)}")
                self.result_queue.put(None)

    @torch.no_grad()
    def _process_single_image(self, image: np.ndarray) -> Dict:
        """Process a single image with all models"""
        try:
            # Preprocess
            tensor_img = self._preprocess_image(image)
            
            # YOLO detection
            yolo_results = self.yolo(tensor_img, verbose=False)[0]
            
            # Classification models
            efficient_pred = self.efficient(tensor_img)
            convnext_pred = self.convnext(tensor_img)
            
            # Ensemble predictions
            ensemble_pred = self._ensemble_predictions(
                yolo_results, efficient_pred, convnext_pred
            )
            
            return {
                'detections': yolo_results.boxes.data.cpu().numpy(),
                'classification': ensemble_pred.cpu().numpy(),
                'confidence': float(ensemble_pred.max())
            }
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Resize
        image = cv2.resize(image, self.config.IMG_SIZE)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # To tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _ensemble_predictions(self, 
                            yolo_results, 
                            efficient_pred: torch.Tensor,
                            convnext_pred: torch.Tensor) -> torch.Tensor:
        """Combine predictions from all models"""
        # Weighted average of classification predictions
        class_pred = 0.4 * F.softmax(efficient_pred, dim=1) + \
                    0.3 * F.softmax(convnext_pred, dim=1)
        
        # Incorporate YOLO confidence
        if len(yolo_results.boxes) > 0:
            yolo_conf = yolo_results.boxes.conf.mean()
            class_pred *= (0.3 * yolo_conf)
            
        return class_pred

    def _warmup(self):
        """Warm up models for consistent inference time"""
        self.logger.info("Warming up models...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        
        for _ in range(self.config.WARM_UP_ITERATIONS):
            self._process_single_image(dummy_input)
            
        self.logger.info("Warmup complete")

    def predict(self, image: np.ndarray, async_mode: bool = True) -> Dict:
        """Public interface for predictions"""
        if async_mode:
            # Async prediction
            self.inference_queue.put(image)
            return self.result_queue.get()
        else:
            # Synchronous prediction
            return self._process_single_image(image)

    def start_video_stream(self, source: int = 0):
        """Process video stream in real-time"""
        cap = cv2.VideoCapture(source)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.predict(frame)
                
                # Visualize results
                self._visualize_results(frame, result)
                
                # Display
                cv2.imshow('Biofouling Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _visualize_results(self, frame: np.ndarray, result: Dict):
        """Visualize detection results on frame"""
        if result is None:
            return
            
        # Draw detections
        for box in result['detections']:
            x1, y1, x2, y2, conf, cls = box
            if conf > self.config.CONFIDENCE_THRESHOLD:
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
        # Add classification result
        class_id = result['classification'].argmax()
        conf = result['confidence']
        cv2.putText(frame, 
                   f"Class: {class_id} ({conf:.2f})",
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0), 
                   2)

class ProductionDeployment:
    """Handles production deployment and monitoring"""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = ProductionModel(config)
        
        # Initialize metrics tracking
        self.metrics = {
            'inference_times': [],
            'accuracy': [],
            'fps': [],
            'memory_usage': []
        }
        
        # Setup monitoring
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Setup real-time monitoring"""
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///production_metrics.db")
        mlflow.start_run(run_name="production_monitoring")

    def _monitor_performance(self):
        """Monitor system performance"""
        while True:
            try:
                # Collect metrics
                metrics = {
                    'gpu_utilization': self._get_gpu_utilization(),
                    'memory_usage': self._get_memory_usage(),
                    'inference_latency': np.mean(self.metrics['inference_times'][-100:]),
                    'fps': self._calculate_fps()
                }
                
                # Log to MLflow
                mlflow.log_metrics(metrics)
                
                # Check for performance degradation
                self._check_performance_alerts(metrics)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.utilization())
            return 0.0
        except:
            return 0.0

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        usage = {
            'system': psutil.virtual_memory().percent,
            'python': psutil.Process().memory_percent()
        }
        
        if torch.cuda.is_available():
            usage['gpu'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        return usage

    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        recent_times = self.metrics['inference_times'][-100:]
        if len(recent_times) < 2:
            return 0.0
        return 1.0 / np.mean(np.diff(recent_times))

    def _check_performance_alerts(self, metrics: Dict):
        """Check and alert for performance issues"""
        alerts = []
        
        # Check FPS
        if metrics['fps'] < 25:  # Target is 30 FPS
            alerts.append(f"Low FPS: {metrics['fps']:.1f}")
            
        # Check latency
        if metrics['inference_latency'] > self.config.MAX_LATENCY_MS:
            alerts.append(f"High latency: {metrics['inference_latency']:.1f}ms")
            
        # Check memory
        if metrics['memory_usage']['python'] > 80:
            alerts.append("High memory usage")
            
        # Log alerts
        if alerts:
            self.logger.warning(f"Performance alerts: {', '.join(alerts)}")

    def start_production(self, video_source: int = 0):
        """Start production system"""
        self.logger.info("Starting production system...")
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                inference_start = time.time()
                result = self.model.predict(frame, async_mode=True)
                inference_time = time.time() - inference_start
                
                # Update metrics
                self.metrics['inference_times'].append(inference_time)
                frame_count += 1
                
                # Visualize results
                self._visualize_production_results(frame, result)
                
                # Display performance metrics
                self._display_metrics(frame)
                
                # Show frame
                cv2.imshow('Production Monitor', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            self.logger.error(f"Production error: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            mlflow.end_run()

    def _visualize_production_results(self, frame: np.ndarray, result: Dict):
        """Visualize results with production metrics"""
        if result is None:
            return
            
        # Draw detections with confidence
        for box in result['detections']:
            x1, y1, x2, y2, conf, cls = box
            if conf > self.config.CONFIDENCE_THRESHOLD:
                # Draw box
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"Class {int(cls)} ({conf:.2f})"
                cv2.putText(frame, 
                           label,
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 255, 0),
                           2)

    def _display_metrics(self, frame: np.ndarray):
        """Display performance metrics on frame"""
        metrics_text = [
            f"FPS: {self._calculate_fps():.1f}",
            f"Latency: {np.mean(self.metrics['inference_times'][-100:]):.1f}ms",
            f"GPU: {self._get_gpu_utilization():.1f}%",
            f"Memory: {self._get_memory_usage()['python']:.1f}%"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame,
                       text,
                       (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 255),
                       2)

def main():
    """Main production entry point"""
    # Initialize configuration
    config = ProductionConfig()
    
    # Create deployment
    deployment = ProductionDeployment(config)
    
    try:
        # Start production system
        deployment.start_production()
    except KeyboardInterrupt:
        logging.info("Production system stopped by user")
    except Exception as e:
        logging.error(f"Production system error: {str(e)}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        logging.info("Production system shutdown complete")

if __name__ == "__main__":
    main()
