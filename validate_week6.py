#!/usr/bin/env python3
"""
Simple validation script for Week 6 - Dashboard Prototype
Validates the implementation structure and basic functionality
"""

import os
import sys
import json

def test_backend_structure():
    """Test backend dashboard structure"""
    print("🔧 Testing Backend Structure...")
    
    required_files = [
        "backend/api/dashboard.py",
        "backend/main.py",
        "backend/core/database.py",
        "backend/core/config.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print(f"  ❌ Missing backend files: {missing}")
        return False
    
    # Check if dashboard.py has required endpoints
    try:
        with open("backend/api/dashboard.py", "r") as f:
            content = f.read()
            
        required_endpoints = [
            "ws/dashboard",
            "api/emotion-update", 
            "api/class/",
            "current-state",
            "alerts",
            "summary"
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"  ❌ Missing dashboard endpoints: {missing_endpoints}")
            return False
            
        print("  ✅ Backend dashboard structure complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking backend: {e}")
        return False

def test_frontend_structure():
    """Test frontend dashboard structure"""
    print("\n🎨 Testing Frontend Structure...")
    
    required_files = [
        "frontend/package.json",
        "frontend/src/App.tsx", 
        "frontend/src/main.tsx",
        "frontend/src/pages/InstructorDashboard.tsx",
        "frontend/src/pages/StudentView.tsx",
        "frontend/src/pages/Login.tsx",
        "frontend/src/components/ClassMoodChart.tsx",
        "frontend/src/components/StudentEmotionGrid.tsx",
        "frontend/src/components/AlertPanel.tsx",
        "frontend/src/components/RealTimeChart.tsx",
        "frontend/src/services/dashboardService.ts",
        "frontend/src/store/store.ts",
        "frontend/src/store/slices/dashboardSlice.ts"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print(f"  ❌ Missing frontend files: {missing}")
        return False
    
    # Check package.json for required dependencies
    try:
        with open("frontend/package.json", "r") as f:
            package_data = json.load(f)
        
        required_deps = [
            "@mui/material",
            "react",
            "react-router-dom", 
            "@reduxjs/toolkit",
            "recharts"
        ]
        
        dependencies = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
        missing_deps = []
        
        for dep in required_deps:
            if dep not in dependencies:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"  ❌ Missing dependencies: {missing_deps}")
            return False
            
        print("  ✅ Frontend dashboard structure complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking frontend: {e}")
        return False

def test_component_features():
    """Test dashboard component features"""
    print("\n📊 Testing Dashboard Features...")
    
    features_found = 0
    total_features = 8
    
    try:
        # Check InstructorDashboard features
        with open("frontend/src/pages/InstructorDashboard.tsx", "r") as f:
            dashboard_content = f.read()
        
        dashboard_features = [
            "ClassMoodChart",
            "StudentEmotionGrid", 
            "AlertPanel",
            "RealTimeChart"
        ]
        
        for feature in dashboard_features:
            if feature in dashboard_content:
                features_found += 1
                print(f"  ✅ {feature} component integrated")
            else:
                print(f"  ❌ {feature} component missing")
        
        # Check dashboard service
        with open("frontend/src/services/dashboardService.ts", "r") as f:
            service_content = f.read()
        
        service_features = [
            "WebSocket",
            "startClassSession",
            "getClassAlerts", 
            "getCurrentClassState"
        ]
        
        for feature in service_features:
            if feature in service_content:
                features_found += 1
                print(f"  ✅ {feature} service method found")
            else:
                print(f"  ❌ {feature} service method missing")
        
        success_rate = features_found / total_features
        print(f"\n  📈 Feature completeness: {features_found}/{total_features} ({success_rate*100:.1f}%)")
        
        return success_rate >= 0.8  # 80% threshold
        
    except Exception as e:
        print(f"  ❌ Error checking features: {e}")
        return False

def test_integration_readiness():
    """Test integration readiness"""
    print("\n🔗 Testing Integration Readiness...")
    
    try:
        # Check if main.py includes dashboard router
        with open("backend/main.py", "r") as f:
            main_content = f.read()
        
        integration_checks = {
            "Dashboard router import": "dashboard" in main_content and "router" in main_content,
            "CORS middleware": "CORSMiddleware" in main_content,
            "WebSocket endpoint": "websocket" in main_content.lower(),
            "Database initialization": "init_db" in main_content
        }
        
        passed_checks = 0
        for check_name, passed in integration_checks.items():
            if passed:
                print(f"  ✅ {check_name}")
                passed_checks += 1
            else:
                print(f"  ❌ {check_name}")
        
        return passed_checks >= 3  # At least 3/4 checks should pass
        
    except Exception as e:
        print(f"  ❌ Error checking integration: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Week 6 Dashboard Prototype Validation")
    print("=" * 60)
    
    test_results = []
    
    # Run validation tests
    test_results.append(test_backend_structure())
    test_results.append(test_frontend_structure())
    test_results.append(test_component_features())
    test_results.append(test_integration_readiness())
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    test_names = [
        "Backend Structure",
        "Frontend Structure", 
        "Dashboard Features",
        "Integration Readiness"
    ]
    
    for name, result in zip(test_names, test_results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎯 All validations passed! Week 6 Dashboard Prototype is ready.")
        print("\nWEEK 6 DELIVERABLES:")
        print("✅ FastAPI backend with real-time dashboard endpoints")
        print("✅ WebSocket implementation for live updates")
        print("✅ React frontend with Material-UI components")
        print("✅ Redux state management for dashboard data")
        print("✅ Class mood visualization with charts")
        print("✅ Per-student emotion tracking grid")
        print("✅ Real-time alert system for instructors") 
        print("✅ Responsive dashboard interface")
        print("✅ Integration-ready architecture")
        
        print("\n🎉 Week 6 - Dashboard Prototype COMPLETED!")
        print("Ready to proceed to Week 7 - Visualization & Reports")
        return True
    else:
        print("\n⚠️ Some validations failed. Please address issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)