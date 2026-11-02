import json
import py_compile
import tempfile
import sys
import re
from pathlib import Path
import nbformat

sys.stdout.reconfigure(encoding="utf-8")

def validate_kaggle_notebook(nb_path: str):
    print(f"=== Validating {nb_path} ===")
    
    # 1. Read & validate with nbformat
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    try:
        nbformat.validate(nb)
        print("✅ nbformat validation: PASSED (Notebook structure is valid v4 JSON)")
    except Exception as e:
        print(f"❌ nbformat validation FAILED: {e}")
        return False
        
    print(f"Total cells: {len(nb.cells)}")
    
    # 2. Extract code cells and compile with py_compile
    code_cell_idx = 0
    errors = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            code_cell_idx += 1
            source = cell.source
            # Strip ipython magic lines like !cmd or %magic
            clean_lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    # replace with pass to preserve line numbers
                    clean_lines.append("# " + line)
                else:
                    clean_lines.append(line)
            clean_code = "\n".join(clean_lines)
            
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(clean_code)
                tmp_path = tmp.name
                
            try:
                py_compile.compile(tmp_path, doraise=True)
                print(f"  Cell {i:2d} (Code Cell #{code_cell_idx:2d}): py_compile ✅ PASSED")
            except py_compile.PyCompileError as pe:
                print(f"  Cell {i:2d} (Code Cell #{code_cell_idx:2d}): py_compile ❌ FAILED:\n{pe}")
                errors += 1
            except Exception as ex:
                print(f"  Cell {i:2d} (Code Cell #{code_cell_idx:2d}): Error ❌ {ex}")
                errors += 1
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
    if errors == 0:
        print("✅ All Python code cells compiled cleanly with py_compile!")
        return True
    else:
        print(f"❌ Found {errors} syntax errors in code cells.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        nb_file = sys.argv[1]
    else:
        nb_file = "maskan3d-lab/notebooks/maskan3d_kaggle.ipynb"
    success = validate_kaggle_notebook(nb_file)
    sys.exit(0 if success else 1)

