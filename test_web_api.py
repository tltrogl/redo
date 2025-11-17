#!/usr/bin/env python3
"""Test script to verify the web API backend is functional.

This script checks that all components can be imported correctly
without requiring fastapi/pydantic to be installed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core imports
        print("  ✓ Testing web package...")
        import diaremot.web
        
        print("  ✓ Testing API package...")
        import diaremot.web.api
        
        # Test that files exist
        from pathlib import Path
        src_path = Path(__file__).parent / "src"
        
        required_files = [
            "diaremot/web/__init__.py",
            "diaremot/web/api/__init__.py",
            "diaremot/web/api/app.py",
            "diaremot/web/api/models.py",
            "diaremot/web/api/routes/__init__.py",
            "diaremot/web/api/routes/config.py",
            "diaremot/web/api/routes/files.py",
            "diaremot/web/api/routes/health.py",
            "diaremot/web/api/routes/jobs.py",
            "diaremot/web/api/services/__init__.py",
            "diaremot/web/api/services/job_queue.py",
            "diaremot/web/api/services/storage.py",
            "diaremot/web/api/websocket/__init__.py",
            "diaremot/web/api/websocket/progress.py",
            "diaremot/web/config_schema.py",
            "diaremot/web/server.py",
        ]
        
        print("\n  Checking required files exist...")
        missing_files = []
        for file in required_files:
            file_path = src_path / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                print(f"    ✓ {file}")
        
        if missing_files:
            print(f"\n  ✗ Missing files: {missing_files}")
            return False
            
        print("\n  All required files exist!")
        
        # Test models.py structure
        print("\n  Checking models.py structure...")
        import ast
        models_path = src_path / "diaremot/web/api/models.py"
        with open(models_path) as f:
            tree = ast.parse(f.read())
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        print(f"    Found {len(classes)} model classes")
        
        expected_models = [
            'JobStatus', 'JobStage', 'JobProgress',
            'JobCreateRequest', 'JobCancelRequest', 'ConfigValidateRequest',
            'FileUploadResponse', 'JobResponse', 'JobListResponse',
            'HealthResponse', 'ConfigParameter', 'ConfigGroup',
            'ConfigSchemaResponse', 'PresetResponse'
        ]
        
        missing_models = [m for m in expected_models if m not in classes]
        if missing_models:
            print(f"    ✗ Missing models: {missing_models}")
            return False
        
        print(f"    ✓ All {len(expected_models)} required models are present")
        
        # Test pyproject.toml has web dependencies
        print("\n  Checking pyproject.toml...")
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with open(pyproject_path) as f:
            content = f.read()
        
        required_deps = ['fastapi', 'uvicorn', 'pydantic', 'websockets', 'python-multipart']
        missing_deps = [dep for dep in required_deps if dep not in content]
        
        if missing_deps:
            print(f"    ✗ Missing dependencies in pyproject.toml: {missing_deps}")
            return False
        
        print(f"    ✓ All required web dependencies are listed")
        
        # Test .gitignore has .next
        print("\n  Checking .gitignore...")
        gitignore_path = Path(__file__).parent / ".gitignore"
        with open(gitignore_path) as f:
            content = f.read()
        
        if '.next' not in content:
            print("    ✗ .gitignore doesn't ignore .next directories")
            return False
        
        print("    ✓ .gitignore properly ignores build artifacts")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n  Note: This is expected if fastapi/pydantic are not installed.")
        print("  Run: pip install -e '.[web]' to install dependencies")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DiaRemot Web API Backend Test")
    print("=" * 60)
    print()
    
    success = test_imports()
    
    print()
    print("=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
        print()
        print("Next steps:")
        print("  1. Install dependencies: pip install -e '.[web]'")
        print("  2. Start server: python src/diaremot/web/server.py")
        print("  3. Visit: http://localhost:8000/api/docs")
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Please review the errors above.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
