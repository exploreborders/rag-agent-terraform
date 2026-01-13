#!/usr/bin/env python3
"""
Integration test runner for the RAG Agent system.
Runs tests with real PostgreSQL and Redis containers using testcontainers.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_integration_tests():
    """Run integration tests with real database containers."""

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("🚀 Starting RAG Agent Integration Tests")
    print("=" * 50)

    # Check if Docker is available
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available. Integration tests require Docker.")
        return 1

    # Set environment for integration tests
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "1"

    # Run integration tests
    test_commands = [
        # Database integration tests
        ["python", "-m", "pytest", "tests/integration/test_database_integration.py", "-v", "--tb=short"],
        # Redis integration tests
        ["python", "-m", "pytest", "tests/integration/test_redis_integration.py", "-v", "--tb=short"],
        # End-to-end integration tests
        ["python", "-m", "pytest", "tests/integration/test_end_to_end.py", "-v", "--tb=short"],
    ]

    all_passed = True

    for i, cmd in enumerate(test_commands, 1):
        print(f"\n📋 Running Test Suite {i}/{len(test_commands)}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 40)

        try:
            result = subprocess.run(
                cmd,
                env=env,
                cwd=project_root / "src",
                capture_output=False,  # Show output live
                text=True
            )

            if result.returncode == 0:
                print(f"✅ Test Suite {i} PASSED")
            else:
                print(f"❌ Test Suite {i} FAILED (exit code: {result.returncode})")
                all_passed = False

        except KeyboardInterrupt:
            print("\n⏹️  Tests interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running test suite {i}: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Database operations validated")
        print("✅ Redis caching validated")
        print("✅ End-to-end workflows validated")
        print("✅ System integration verified")
        return 0
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        print("🔍 Check the output above for details")
        return 1

def run_quick_integration_test():
    """Run a quick integration test to verify basic functionality."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("🔍 Running Quick Integration Test")

    # Run just one test from each suite to verify setup works
    quick_tests = [
        "tests/integration/test_database_integration.py::TestDatabaseIntegration::test_vector_store_connection",
        "tests/integration/test_redis_integration.py::TestRedisIntegration::test_redis_connection",
        "tests/integration/test_end_to_end.py::TestEndToEndIntegration::test_complete_document_processing_workflow"
    ]

    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "1"

    all_passed = True

    for test_path in quick_tests:
        print(f"\n🧪 Testing: {test_path}")

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                env=env,
                cwd=project_root / "src",
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print(f"✅ {test_path} PASSED")
            else:
                print(f"❌ {test_path} FAILED")
                all_passed = False

        except Exception as e:
            print(f"❌ Error running {test_path}: {e}")
            all_passed = False

    return 0 if all_passed else 1

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        exit_code = run_quick_integration_test()
    else:
        exit_code = run_integration_tests()

    sys.exit(exit_code)</content>
<parameter name="filePath">run_integration_tests.py