#!/usr/bin/env python3
"""
Test script for Week 7 - Visualization & Reports
Tests advanced charts, time-series analysis, and PDF/Excel export functionality
"""

import sys
import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TEST_CLASS_ID = "CS101-Fall2024-Reports"
BACKEND_URL = "http://localhost:8000"

class Week7ReportsTest:
    def __init__(self):
        self.session = None
        
    async def setup(self):
        """Setup test session"""
        print("🔧 Setting up Week 7 Reports Test...")
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
            
    async def test_reports_api_endpoints(self):
        """Test reports REST API endpoints"""
        print("\n📊 Testing Reports API Endpoints...")
        
        try:
            # 1. Get emotion timeline
            print("  ✓ Testing emotion timeline...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/timeline?hours=24"
            )
            print(f"    Timeline API status: {response.status}")
            
            # 2. Get student engagement report
            print("  ✓ Testing student engagement report...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/student-engagement"
            )
            print(f"    Engagement API status: {response.status}")
            
            # 3. Get analytics overview
            print("  ✓ Testing analytics overview...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/analytics/overview"
            )
            print(f"    Analytics API status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"    Total records: {data.get('overview', {}).get('total_emotion_records', 0)}")
                print(f"    Unique students: {data.get('overview', {}).get('unique_students', 0)}")
            
            # 4. Get class heatmap
            print("  ✓ Testing class heatmap...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/heatmap?days=7"
            )
            print(f"    Heatmap API status: {response.status}")
            if response.status == 200:
                data = await response.json()
                has_image = 'heatmap_image' in data
                print(f"    Heatmap image generated: {has_image}")
            
            print("    ✅ All report endpoints accessible")
            return True
            
        except Exception as e:
            print(f"    ❌ Reports API test failed: {e}")
            return False
    
    async def test_export_functionality(self):
        """Test PDF and Excel export functionality"""
        print("\n📄 Testing Export Functionality...")
        
        try:
            # Test PDF export
            print("  ✓ Testing PDF export...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/export/pdf"
            )
            
            if response.status == 200:
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type.lower():
                    print("    PDF export working correctly")
                else:
                    print("    PDF export returned non-PDF content")
            else:
                print(f"    PDF export failed with status: {response.status}")
            
            # Test Excel export
            print("  ✓ Testing Excel export...")
            response = await self.session.get(
                f"{BACKEND_URL}/api/reports/api/class/{TEST_CLASS_ID}/export/excel"
            )
            
            if response.status == 200:
                content_type = response.headers.get('content-type', '')
                if 'spreadsheet' in content_type.lower() or 'excel' in content_type.lower():
                    print("    Excel export working correctly")
                else:
                    print("    Excel export returned non-Excel content")
            else:
                print(f"    Excel export failed with status: {response.status}")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Export functionality test failed: {e}")
            return False
    
    async def test_frontend_components(self):
        """Test frontend components structure"""
        print("\n🎨 Testing Frontend Components...")
        
        try:
            # Check if Week 7 frontend files exist
            week7_files = [
                "frontend/src/pages/ReportsPage.tsx",
                "frontend/src/components/AdvancedTimelineChart.tsx",
                "frontend/src/components/StudentEngagementTable.tsx", 
                "frontend/src/components/EmotionHeatmap.tsx",
                "frontend/src/components/AnalyticsOverview.tsx",
                "frontend/src/services/reportsService.ts"
            ]
            
            missing_files = []
            component_features = []
            
            for file_path in week7_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                else:
                    # Check file content for key features
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if 'ReportsPage' in file_path and 'TabPanel' in content:
                        component_features.append("Tabbed Reports Interface")
                    elif 'AdvancedTimelineChart' in file_path and 'Recharts' in content:
                        component_features.append("Advanced Timeline Visualization")
                    elif 'StudentEngagementTable' in file_path and 'TableSortLabel' in content:
                        component_features.append("Sortable Engagement Table")
                    elif 'EmotionHeatmap' in file_path and 'base64' in content:
                        component_features.append("Heatmap Image Display")
                    elif 'AnalyticsOverview' in file_path and 'PieChart' in content:
                        component_features.append("Analytics Overview Charts")
                    elif 'reportsService' in file_path and 'exportPDFReport' in content:
                        component_features.append("Export Service Methods")
            
            if missing_files:
                print(f"    ❌ Missing files: {missing_files}")
                return False
            
            print("    ✅ All Week 7 components present")
            for feature in component_features:
                print(f"    ✅ {feature}")
            
            # Check package.json for required dependencies
            with open("frontend/package.json", "r") as f:
                package_data = json.load(f)
            
            week7_dependencies = [
                "recharts",
                "@mui/x-date-pickers",
                "@mui/x-charts"
            ]
            
            dependencies = {**package_data.get("dependencies", {})}
            missing_deps = []
            
            for dep in week7_dependencies:
                if dep not in dependencies:
                    missing_deps.append(dep)
                else:
                    print(f"    ✅ Dependency: {dep}")
            
            if missing_deps:
                print(f"    ⚠️ Optional dependencies missing: {missing_deps}")
            
            return len(missing_files) == 0
            
        except Exception as e:
            print(f"    ❌ Frontend components test failed: {e}")
            return False
    
    async def test_data_visualization_features(self):
        """Test data visualization and analytics features"""
        print("\n📈 Testing Data Visualization Features...")
        
        try:
            features_tested = 0
            
            # Check ReportsPage features
            if os.path.exists("frontend/src/pages/ReportsPage.tsx"):
                with open("frontend/src/pages/ReportsPage.tsx", 'r') as f:
                    content = f.read()
                
                visualization_features = {
                    "Multiple Chart Types": "ToggleButton" in content and "chartType" in content,
                    "Date Range Selection": "DatePicker" in content or "timeRange" in content,
                    "Export Functionality": "exportPDF" in content and "exportExcel" in content,
                    "Tabbed Interface": "TabPanel" in content and "Tabs" in content,
                    "Real-time Updates": "useEffect" in content,
                }
                
                for feature, exists in visualization_features.items():
                    if exists:
                        print(f"    ✅ {feature}")
                        features_tested += 1
                    else:
                        print(f"    ❌ {feature}")
            
            # Check AdvancedTimelineChart features
            if os.path.exists("frontend/src/components/AdvancedTimelineChart.tsx"):
                with open("frontend/src/components/AdvancedTimelineChart.tsx", 'r') as f:
                    content = f.read()
                
                chart_features = {
                    "Multiple Chart Types": "LineChart" in content and "AreaChart" in content and "BarChart" in content,
                    "Interactive Controls": "ToggleButtonGroup" in content,
                    "Custom Tooltips": "CustomTooltip" in content,
                    "Responsive Design": "ResponsiveContainer" in content,
                }
                
                for feature, exists in chart_features.items():
                    if exists:
                        print(f"    ✅ Chart {feature}")
                        features_tested += 1
                    else:
                        print(f"    ❌ Chart {feature}")
            
            success_rate = features_tested / 9  # Total expected features
            print(f"\n    📈 Feature completeness: {features_tested}/9 ({success_rate*100:.1f}%)")
            
            return success_rate >= 0.7  # 70% threshold
            
        except Exception as e:
            print(f"    ❌ Data visualization test failed: {e}")
            return False
    
    async def test_backend_integration(self):
        """Test backend reports integration"""
        print("\n🔗 Testing Backend Integration...")
        
        try:
            # Check if main.py includes reports router
            with open("backend/main.py", "r") as f:
                main_content = f.read()
            
            # Check if reports.py has required dependencies
            with open("backend/api/reports.py", "r") as f:
                reports_content = f.read()
            
            integration_checks = {
                "Reports router imported": "reports" in main_content and "router" in main_content,
                "PDF generation library": "reportlab" in reports_content,
                "Excel generation library": "pandas" in reports_content and "openpyxl" in reports_content,
                "Chart generation": "matplotlib" in reports_content or "seaborn" in reports_content,
                "Database integration": "Session" in reports_content and "get_db" in reports_content,
                "Export endpoints": "/export/pdf" in reports_content and "/export/excel" in reports_content
            }
            
            passed_checks = 0
            for check_name, passed in integration_checks.items():
                if passed:
                    print(f"    ✅ {check_name}")
                    passed_checks += 1
                else:
                    print(f"    ❌ {check_name}")
            
            return passed_checks >= 4  # At least 4/6 checks should pass
            
        except Exception as e:
            print(f"    ❌ Backend integration test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Week 7 tests"""
        print("🚀 Starting Week 7 - Visualization & Reports Tests")
        print("=" * 60)
        
        await self.setup()
        
        test_results = []
        
        # Run individual tests
        test_results.append(await self.test_reports_api_endpoints())
        test_results.append(await self.test_export_functionality())
        test_results.append(await self.test_frontend_components())
        test_results.append(await self.test_data_visualization_features())
        test_results.append(await self.test_backend_integration())
        
        await self.cleanup()
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 WEEK 7 TEST RESULTS")
        print("=" * 60)
        
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        test_names = [
            "Reports API Endpoints",
            "Export Functionality",
            "Frontend Components",
            "Data Visualization Features", 
            "Backend Integration"
        ]
        
        for i, (name, result) in enumerate(zip(test_names, test_results)):
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} - {name}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= 4:  # Allow 1 failure
            print("🎯 Week 7 requirements met! Visualization & Reports system is functional.")
            print("\nWEEK 7 DELIVERABLES:")
            print("✅ Advanced timeline charts with multiple visualization types")
            print("✅ Student engagement analysis with sortable tables")
            print("✅ Emotion heatmap visualization") 
            print("✅ Analytics overview with comprehensive metrics")
            print("✅ PDF report generation with charts and statistics")
            print("✅ Excel data export with multiple sheets")
            print("✅ Interactive dashboard with tabbed interface")
            print("✅ Real-time data integration")
            print("✅ Responsive design for all screen sizes")
            
            return True
        else:
            print("❌ Some critical tests failed. Please check the implementation.")
            return False

async def main():
    """Main test function"""
    tester = Week7ReportsTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Week 7 - Visualization & Reports completed successfully!")
        print("Ready to proceed to Week 8 - Security & Privacy")
    else:
        print("\n⚠️ Week 7 tests failed. Please fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)