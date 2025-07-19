#!/usr/bin/env python3
"""
Simple Go2 Vision Test Script

Tests Go2 environment with vision enabled and verifies image capture.
"""

import os
import jax
import numpy as np
import rerun as rr
from PIL import Image

# GPU memory management and precision
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = str(2**31)

from mujoco_playground import registry

def save_image_to_disk(img_array, filename, output_dir="go2_vision_output"):
    """Save numpy array as image to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure image is in the right format (0-255, uint8)
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            # Assume values are in [0, 1] range
            img_array = (img_array * 255).astype(np.uint8)
        else:
            # Assume values are already in [0, 255] range
            img_array = img_array.astype(np.uint8)
    
    # Create PIL image and save
    pil_image = Image.fromarray(img_array)
    filepath = os.path.join(output_dir, filename)
    pil_image.save(filepath)
    print(f"💾 Saved image to: {filepath}")
    return filepath

def test_go2_vision():
    """Test Go2 environment with vision enabled."""
    print("🤖 Testing Go2 Vision Setup...")
    
    rr.init("go2", spawn=True)
    
    print("\nTesting basic Go2 environment...")
    try:
        # Try with minimal config
        env = registry.load("Go2JoystickFlatTerrain", config_overrides={"episode_length": 5})
        print("✓ Basic Go2 environment loaded")
        
        key = jax.random.PRNGKey(0)
        print("About to call env.reset()...")
        state = env.reset(key)
        print("env.reset() returned!")
        print("✓ Basic environment reset successful")
        print(f"Basic obs keys: {list(state.obs.keys())}")
        
    except Exception as e:
        print(f"❌ Basic environment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now test with vision
    print("\nTesting with vision...")
    config_overrides = {
        "vision": True,
        "vision_config.render_width": 32,  # Smaller to start
        "vision_config.render_height": 32,
        # Use the MJX raytracer backend for stability (avoid rasterizer issues)
        "vision_config.use_rasterizer": True,
        "episode_length": 10,
    }
    
    print("Loading Go2 environment with vision...")
    try:
        env = registry.load("Go2JoystickFlatTerrain", config_overrides=config_overrides)
        print(f"✓ Environment loaded - has vision: {hasattr(env, '_vision') and env._vision}")
    except Exception as e:
        print(f"❌ Vision environment loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if renderer was created
    if hasattr(env, 'renderer'):
        print("✓ Madrona renderer initialized")
        print(f"  - Camera ID: {env.renderer.enabled_cameras}")
        print(f"  - Render size: {env._config.vision_config.render_width}x{env._config.vision_config.render_height}")
    else:
        print("⚠️  No renderer found")
        return
    
    print("Resetting environment (this may take a moment)...")
    key = jax.random.PRNGKey(0)
    
    try:
        print("Calling env.reset...")
        state = env.reset(key)
        print("✓ Environment reset successful!")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check observations
    print(f"Observation keys: {list(state.obs.keys()) if hasattr(state.obs, 'keys') else 'No vision obs'}")
    
    if hasattr(state.obs, 'keys') and 'pixels/front_camera' in state.obs:
        vision_obs = state.obs['pixels/front_camera']
        print(f"✓ Vision observation shape: {vision_obs.shape}")
        
        # Handle both batched and single environment cases
        if len(vision_obs.shape) == 4:  # Batched
            print(f"  - Batch size: {vision_obs.shape[0]}")
            print(f"  - Image dimensions: {vision_obs.shape[1]}x{vision_obs.shape[2]}")
            print(f"  - Channels: {vision_obs.shape[3]}")
            img = np.array(vision_obs[0])  # First environment
        else:  # Single environment
            print(f"  - Image dimensions: {vision_obs.shape[0]}x{vision_obs.shape[1]}")
            print(f"  - Channels: {vision_obs.shape[2]}")
            img = np.array(vision_obs)
            
        # Handle stacked frames (take first 3 channels for RGB)
        if img.shape[-1] > 3:
            rgb_img = img[..., :3]  # Take first 3 channels
            print(f"  - Using RGB channels from stacked frames: {rgb_img.shape}")
        else:
            rgb_img = img
        
        # Log the RGB image to Rerun
        rr.log("go2/camera", rr.Image(rgb_img))
        
        # Save the RGB image to disk
        save_image_to_disk(rgb_img, "go2_front_camera_initial.png")
       
        if 'state' in state.obs:
            robot_state = np.array(state.obs['state'])
            print(f"  - Robot state shape: {robot_state.shape}")
        
        print("✓ Successfully captured and logged image from Go2 front camera to Rerun!")
        # Vision capture succeeded; skip further steps in this example to avoid runtime issues
        print("🎉 Go2 vision test completed successfully!")
        print("💡 Check the Rerun viewer to see the logged data!")

    else:
        print("❌ No vision observations found. Check madrona-mjx installation.")
        return
    
    # Test a few steps and log to Rerun
    print("Testing environment steps...")
    for i in range(10):  # More steps for better visualization
        action = jax.random.uniform(
            jax.random.PRNGKey(i), 
            (env.action_size,), 
            minval=-0.5, maxval=0.5
        )
        state = env.step(state, action)
        
        # Log vision if available
        if hasattr(state.obs, 'keys') and 'pixels/front_camera' in state.obs:
            vision_obs = state.obs['pixels/front_camera']
            if len(vision_obs.shape) == 4:
                img = np.array(vision_obs[0])
            else:
                img = np.array(vision_obs)
            
            if img.shape[-1] > 3:
                rgb_img = img[..., :3]
            else:
                rgb_img = img
            
            rr.log("go2/camera", rr.Image(rgb_img))
            
            save_image_to_disk(rgb_img, f"go2_camera_step_{i:03d}.png")
        
        # Log robot state
        if 'state' in state.obs:
            robot_state = np.array(state.obs['state'])
        
        print(f"Step {i+1}: Reward = {state.reward:.3f}")
    
    print("🎉 Go2 vision test completed successfully!")
    print("💡 Check the Rerun viewer to see the logged data!")

if __name__ == "__main__":
    test_go2_vision()