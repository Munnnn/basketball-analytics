"""
Model path management
"""

import os
from typing import Dict, Optional
from pathlib import Path


class ModelPaths:
    """Manage paths to ML models"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        
        # Default model paths
        self.models = {
            'yolo': {
                'default': 'yolo/best.pt',
                'basketball_v1': 'yolo/basketball_v1.pt',
                'basketball_v2': 'yolo/basketball_v2.pt'
            },
            'sam': {
                'default': 'sam/sam_vit_h_4b8939.pth',
                'edgetam': 'sam/edgetam.pt'
            },
            'siglip': {
                'default': 'siglip/siglip-base-patch16-224'
            },
            'openpose': {
                'default': 'openpose/pose_iter_440000.caffemodel'
            }
        }
        
    def get_model_path(self, model_type: str, version: str = 'default') -> str:
        """
        Get full path to model
        
        Args:
            model_type: Type of model (yolo, sam, etc.)
            version: Model version
            
        Returns:
            Full path to model file
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if version not in self.models[model_type]:
            # Try default
            version = 'default'
            
        relative_path = self.models[model_type][version]
        full_path = self.base_dir / relative_path
        
        # Check if file exists
        if not full_path.exists():
            # Try alternative locations
            alternatives = [
                Path(relative_path),  # Current directory
                Path(f"./checkpoints/{relative_path}"),  # Checkpoints dir
                Path(f"/models/{relative_path}"),  # System models
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    return str(alt_path)
                    
            # Return expected path even if not found
            return str(full_path)
            
        return str(full_path)
    
    def register_model(self, model_type: str, version: str, path: str):
        """Register a new model path"""
        if model_type not in self.models:
            self.models[model_type] = {}
        self.models[model_type][version] = path
        
    def list_models(self) -> Dict[str, Dict[str, str]]:
        """List all registered models"""
        result = {}
        
        for model_type, versions in self.models.items():
            result[model_type] = {}
            for version, path in versions.items():
                full_path = self.base_dir / path
                result[model_type][version] = {
                    'path': str(full_path),
                    'exists': full_path.exists()
                }
                
        return result


# Global model paths instance
_model_paths: Optional[ModelPaths] = None


def get_model_path(model_type: str, version: str = 'default') -> str:
    """Get path to model"""
    global _model_paths
    
    if _model_paths is None:
        base_dir = os.environ.get("BASKETBALL_MODEL_DIR", "./models")
        _model_paths = ModelPaths(base_dir)
        
    return _model_paths.get_model_path(model_type, version)
