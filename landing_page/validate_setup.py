#!/usr/bin/env python3
"""
Validation script to ensure landing page is properly configured.
"""

import json
from pathlib import Path
import sys

def validate_json_file(filepath, expected_keys=None):
    """Validate a JSON file exists and has expected structure."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if expected_keys:
            missing_keys = [k for k in expected_keys if k not in data]
            if missing_keys:
                return False, f"Missing expected keys: {missing_keys}"
        
        return True, f"Valid JSON with {len(data)} entries"
    except FileNotFoundError:
        return False, "File not found"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    landing_page_dir = Path(__file__).parent
    
    print("=" * 60)
    print("LANDING PAGE VALIDATION")
    print("=" * 60)
    print()
    
    checks = []
    
    # Check data files
    print("📊 Data Files:")
    htl_file = landing_page_dir / "hide_label_data.json"
    valid, msg = validate_json_file(htl_file)
    status = "✅" if valid else "❌"
    print(f"  {status} hide_label_data.json: {msg}")
    checks.append(valid)
    
    if valid:
        with open(htl_file) as f:
            htl_data = json.load(f)
        required_opts = ["BO_GP_EI", "SBO_GP_PV", "RANDOM"]
        for opt in required_opts:
            if opt in htl_data:
                print(f"     ✓ {opt}: {htl_data[opt]['mean_steps']:.1f} mean steps")
            else:
                print(f"     ✗ Missing {opt}")
                checks.append(False)
    
    or_file = landing_page_dir / "open_race_data.json"
    valid, msg = validate_json_file(or_file)
    status = "✅" if valid else "❌"
    print(f"  {status} open_race_data.json: {msg}")
    checks.append(valid)
    
    if valid:
        with open(or_file) as f:
            or_data = json.load(f)
        required_opts = ["BO_GP_EI", "SBO_GP_PV", "RANDOM"]
        for opt in required_opts:
            if opt in or_data:
                final_val = or_data[opt]['values'][-1] if or_data[opt].get('values') else 0
                print(f"     ✓ {opt}: {final_val:.2f} final best")
            else:
                print(f"     ✗ Missing {opt}")
                checks.append(False)
    
    print()
    
    # Check HTML files
    print("📄 HTML Files:")
    for filename in ["index.html", "playground.html"]:
        filepath = landing_page_dir / filename
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}: {'Found' if exists else 'Missing'}")
        checks.append(exists)
    
    print()
    
    # Check JavaScript
    print("📜 JavaScript:")
    js_file = landing_page_dir / "script.js"
    exists = js_file.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} script.js: {'Found' if exists else 'Missing'}")
    checks.append(exists)
    
    if exists:
        with open(js_file, 'r') as f:
            js_content = f.read()
        
        # Check for key modifications
        has_load_data = "loadRealData" in js_content
        has_hide_data = "hideLabelData" in js_content
        has_open_data = "openRaceData" in js_content
        
        print(f"     {'✓' if has_load_data else '✗'} loadRealData function present")
        print(f"     {'✓' if has_hide_data else '✗'} hideLabelData variable present")
        print(f"     {'✓' if has_open_data else '✗'} openRaceData variable present")
        
        checks.extend([has_load_data, has_hide_data, has_open_data])
    
    print()
    
    # Check CSS
    print("🎨 Styling:")
    css_file = landing_page_dir / "style.css"
    exists = css_file.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} style.css: {'Found' if exists else 'Missing'}")
    checks.append(exists)
    
    print()
    
    # Check processing script
    print("⚙️  Processing Script:")
    process_file = landing_page_dir / "process_results.py"
    exists = process_file.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} process_results.py: {'Found' if exists else 'Missing'}")
    checks.append(exists)
    
    print()
    
    # Check key results
    print("🎯 Key Results Verification:")
    if 'htl_data' in locals() and 'or_data' in locals():
        # Check Hide-the-Label rankings
        htl_sorted = sorted(htl_data.items(), key=lambda x: x[1]['mean_steps'])
        print(f"  Hide-the-Label Top 3:")
        for i, (opt, stats) in enumerate(htl_sorted[:3], 1):
            marker = " 🏆" if opt in ["BO_GP_EI", "SBO_GP_PV"] else ""
            print(f"    {i}. {opt}: {stats['mean_steps']:.1f} steps{marker}")
        
        # Check Open Race rankings
        or_sorted = sorted(
            [(opt, data['values'][-1] if data.get('values') else 0) 
             for opt, data in or_data.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        print(f"  Open Race Top 3:")
        for i, (opt, val) in enumerate(or_sorted[:3], 1):
            marker = " 🏆" if opt in ["BO_GP_EI", "SBO_GP_PV"] else ""
            print(f"    {i}. {opt}: {val:.2f}{marker}")
        
        # Check that BO methods beat Random
        if "RANDOM" in htl_data and "BO_GP_EI" in htl_data:
            bo_steps = htl_data["BO_GP_EI"]["mean_steps"]
            random_steps = htl_data["RANDOM"]["mean_steps"]
            improvement = (random_steps / bo_steps)
            print(f"\n  ✅ BO_GP_EI is {improvement:.1f}x better than RANDOM in HTL")
            checks.append(improvement > 5)  # Should be at least 5x better
        
    print()
    
    # Final summary
    print("=" * 60)
    all_passed = all(checks)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print()
        print("Your landing page is ready! To view it:")
        print(f"  cd {landing_page_dir}")
        print("  ./view_landing_page.sh")
        print("  # Then open http://localhost:8000")
    else:
        print(f"❌ {len([c for c in checks if not c])} CHECKS FAILED")
        print()
        print("Please review the errors above and fix them.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()

