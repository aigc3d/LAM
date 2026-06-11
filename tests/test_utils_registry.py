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
Tests for lam.utils.registry module.
"""

import pytest
from lam.utils.registry import Registry


class TestRegistry:
    """Tests for the Registry class."""

    def test_registry_initialization(self):
        """Test that Registry initializes with empty registry."""
        registry = Registry()
        assert registry._registry == {}

    def test_register_class(self):
        """Test registering a class with the registry."""
        registry = Registry()

        @registry.register("test_class")
        class TestClass:
            pass

        assert "test_class" in registry
        assert registry["test_class"] is TestClass

    def test_register_function(self):
        """Test registering a function with the registry."""
        registry = Registry()

        @registry.register("test_func")
        def test_function():
            return "hello"

        assert "test_func" in registry
        assert registry["test_func"] is test_function

    def test_register_duplicate_raises_error(self):
        """Test that registering a duplicate name raises AssertionError."""
        registry = Registry()

        @registry.register("duplicate")
        class FirstClass:
            pass

        with pytest.raises(AssertionError, match="Module duplicate already registered"):
            @registry.register("duplicate")
            class SecondClass:
                pass

    def test_get_unregistered_raises_error(self):
        """Test that getting an unregistered name raises KeyError."""
        registry = Registry()

        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_contains_registered(self):
        """Test __contains__ returns True for registered items."""
        registry = Registry()

        @registry.register("exists")
        class ExistingClass:
            pass

        assert "exists" in registry

    def test_contains_unregistered(self):
        """Test __contains__ returns False for unregistered items."""
        registry = Registry()
        assert "nonexistent" not in registry

    def test_multiple_registrations(self):
        """Test registering multiple items."""
        registry = Registry()

        @registry.register("class_a")
        class ClassA:
            pass

        @registry.register("class_b")
        class ClassB:
            pass

        @registry.register("func_c")
        def func_c():
            pass

        assert len(registry._registry) == 3
        assert registry["class_a"] is ClassA
        assert registry["class_b"] is ClassB
        assert registry["func_c"] is func_c

    def test_decorator_returns_original_class(self):
        """Test that the register decorator returns the original class unchanged."""
        registry = Registry()

        @registry.register("my_class")
        class MyClass:
            value = 42

        # The decorator should return the original class
        assert MyClass.value == 42
        instance = MyClass()
        assert isinstance(instance, MyClass)
