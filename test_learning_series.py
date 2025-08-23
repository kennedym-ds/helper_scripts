#!/usr/bin/env python3
"""
Test script for the Machine Learning Learning Series

This script tests the basic functionality of the learning series setup
and ensures all modules can be imported and used correctly.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_directory_structure():
    """Test that all required directories exist"""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'learning_series',
        'learning_series/00_setup',
        'learning_series/01_ml_fundamentals', 
        'learning_series/02_data_preprocessing',
        'learning_series/03_supervised_learning',
        'learning_series/04_unsupervised_learning',
        'learning_series/05_computer_vision',
        'learning_series/06_deep_learning',
        'learning_series/07_generative_ai',
        'learning_series/08_advanced_topics',
        'learning_series/datasets',
        'learning_series/utils',
        'learning_series/projects'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} - MISSING")
            return False
    
    return True

def test_file_existence():
    """Test that key files exist"""
    print("\n📄 Testing key files...")
    
    required_files = [
        'learning_series/README.md',
        'learning_series/requirements.txt',
        'learning_series/00_setup/README.md',
        'learning_series/00_setup/01_environment_setup.ipynb',
        'learning_series/02_data_preprocessing/05_automated_eda.ipynb',
        'learning_series/utils/__init__.py',
        'learning_series/projects/README.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            return False
    
    return True

def test_helper_script_integration():
    """Test integration with existing helper scripts"""
    print("\n🔧 Testing helper script integration...")
    
    try:
        # Test if we can import the utility functions
        sys.path.append('learning_series')
        from utils import generate_classification_dataset, print_dataset_info
        
        # Test utility functions
        X, y, features = generate_classification_dataset(n_samples=50, n_features=5)
        print(f"✅ Generated test dataset: {X.shape}")
        
        print_dataset_info(X, y, "Test Dataset")
        print("✅ Dataset info function working")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper script integration failed: {e}")
        return False

def test_readme_content():
    """Test that README files have substantial content"""
    print("\n📖 Testing README content...")
    
    readme_files = [
        'learning_series/README.md',
        'learning_series/00_setup/README.md',
        'learning_series/05_computer_vision/README.md',
        'learning_series/06_deep_learning/README.md',
        'learning_series/07_generative_ai/README.md'
    ]
    
    for readme_path in readme_files:
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 1000:  # At least 1000 characters
                    print(f"✅ {readme_path} - {len(content)} characters")
                else:
                    print(f"⚠️  {readme_path} - Only {len(content)} characters (might be too short)")
        else:
            print(f"❌ {readme_path} - MISSING")
            return False
    
    return True

def test_notebook_structure():
    """Test that notebooks have proper JSON structure"""
    print("\n📓 Testing notebook structure...")
    
    notebook_files = [
        'learning_series/00_setup/01_environment_setup.ipynb',
        'learning_series/02_data_preprocessing/05_automated_eda.ipynb'
    ]
    
    for notebook_path in notebook_files:
        try:
            import json
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)
                
            # Check for required notebook fields
            required_fields = ['cells', 'metadata', 'nbformat']
            for field in required_fields:
                if field not in notebook_content:
                    print(f"❌ {notebook_path} - Missing field: {field}")
                    return False
            
            # Check that notebook has cells
            if len(notebook_content['cells']) > 5:
                print(f"✅ {notebook_path} - {len(notebook_content['cells'])} cells")
            else:
                print(f"⚠️  {notebook_path} - Only {len(notebook_content['cells'])} cells")
            
        except Exception as e:
            print(f"❌ {notebook_path} - Invalid JSON: {e}")
            return False
    
    return True

def generate_summary_report():
    """Generate a summary of the learning series"""
    print("\n📊 LEARNING SERIES SUMMARY")
    print("=" * 50)
    
    # Count modules
    module_dirs = [d for d in os.listdir('learning_series') if d.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8'))]
    print(f"📚 Total Modules: {len(module_dirs)}")
    
    # Count README files
    readme_count = 0
    for root, dirs, files in os.walk('learning_series'):
        readme_count += sum(1 for f in files if f == 'README.md')
    print(f"📄 README Files: {readme_count}")
    
    # Count notebook files
    notebook_count = 0
    for root, dirs, files in os.walk('learning_series'):
        notebook_count += sum(1 for f in files if f.endswith('.ipynb'))
    print(f"📓 Jupyter Notebooks: {notebook_count}")
    
    # Learning path overview
    print(f"\n🎯 Learning Path Overview:")
    modules = [
        "Module 0: Setup and Python Basics",
        "Module 1: ML Fundamentals", 
        "Module 2: Data Preprocessing (integrates auto_eda.py)",
        "Module 3: Supervised Learning",
        "Module 4: Unsupervised Learning", 
        "Module 5: Computer Vision",
        "Module 6: Deep Learning",
        "Module 7: Generative AI",
        "Module 8: Advanced Topics & MLOps"
    ]
    
    for i, module in enumerate(modules, 1):
        print(f"   {i}. {module}")
    
    print(f"\n🚀 Estimated Total Duration: 40-50 days (120-150 hours)")
    print(f"🎓 Comprehensive coverage from basics to advanced generative AI")

def main():
    """Run all tests and generate report"""
    print("🧪 MACHINE LEARNING LEARNING SERIES TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("File Existence", test_file_existence),
        ("Helper Script Integration", test_helper_script_integration),
        ("README Content", test_readme_content),
        ("Notebook Structure", test_notebook_structure)
    ]
    
    results = []
    for test_name, test_function in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            result = test_function()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n🏁 TEST RESULTS SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Learning series is ready!")
        generate_summary_report()
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)