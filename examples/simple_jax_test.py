import os
import jax
import jax.numpy as jnp

# Set conservative GPU memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEMORY_FRACTION'] = '0.5'

print("Testing basic JAX functionality...")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Test basic JAX operations
try:
    # Simple matrix operations
    a = jnp.ones((100, 100))
    b = jnp.ones((100, 100))
    c = jnp.dot(a, b)
    print("✓ Basic matrix operations work")
    
    # Test JIT compilation
    @jax.jit
    def simple_fn(x):
        return jnp.sum(x * x)
    
    result = simple_fn(jnp.ones(1000))
    print("✓ JIT compilation works")
    
    # Test larger operations that might trigger cuSolver
    large_matrix = jnp.random.normal(size=(500, 500))
    try:
        # This might trigger cuSolver
        eigenvals = jnp.linalg.eigvals(large_matrix)
        print("✓ Linear algebra operations work")
    except Exception as e:
        print(f"✗ Linear algebra failed: {e}")
    
    # Test with the actual environment setup
    print("\nTesting environment setup...")
    from mujoco_playground import registry
    
    env_name = 'Go2JoystickFlatTerrain'
    env = registry.load(env_name)
    print("✓ Environment loaded successfully")
    
    # Test environment reset
    try:
        state = env.reset(jax.random.PRNGKey(0))
        print("✓ Environment reset works")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
    
    # Test environment step
    try:
        action = jnp.zeros(env.action_size)
        state = env.step(state, action)
        print("✓ Environment step works")
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
    
except Exception as e:
    print(f"✗ Basic JAX test failed: {e}")

print("\nTest completed.") 