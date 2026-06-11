# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Basic tests that don't require heavy dependencies.
These tests verify the test infrastructure is working correctly.
"""

import pytest
import sys
import os


class TestBasicInfrastructure:
    """Basic tests to verify test infrastructure."""

    def test_python_version(self):
        """Test Python version is 3.8+."""
        assert sys.version_info >= (3, 8)

    def test_project_structure(self):
        """Test project directory structure exists."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Check main directories exist
        assert os.path.isdir(os.path.join(project_root, "lam"))
        assert os.path.isdir(os.path.join(project_root, "lam", "models"))
        assert os.path.isdir(os.path.join(project_root, "lam", "utils"))
        assert os.path.isdir(os.path.join(project_root, "lam", "losses"))
        assert os.path.isdir(os.path.join(project_root, "lam", "datasets"))

    def test_config_files_exist(self):
        """Test configuration files exist."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        assert os.path.isfile(os.path.join(project_root, "requirements.txt"))
        assert os.path.isfile(os.path.join(project_root, "pytest.ini"))

    def test_imports_structure(self):
        """Test that the lam package is importable."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # This should not raise ImportError even without torch
        # as it just tests the package structure
        assert os.path.isfile(os.path.join(project_root, "lam", "__init__.py"))


class TestRegistry:
    """Test Registry class without torch dependency."""

    def test_registry_basic(self):
        """Test Registry class can be imported and used."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from lam.utils.registry import Registry

        registry = Registry()

        @registry.register("test_item")
        class TestItem:
            pass

        assert "test_item" in registry
        assert registry["test_item"] is TestItem

    def test_registry_multiple_items(self):
        """Test Registry with multiple items."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from lam.utils.registry import Registry

        registry = Registry()

        @registry.register("a")
        class A:
            pass

        @registry.register("b")
        class B:
            pass

        assert len(registry._registry) == 2
        assert "a" in registry
        assert "b" in registry

    def test_registry_duplicate_raises(self):
        """Test Registry raises on duplicate registration."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from lam.utils.registry import Registry

        registry = Registry()

        @registry.register("duplicate")
        class First:
            pass

        with pytest.raises(AssertionError):
            @registry.register("duplicate")
            class Second:
                pass
