import os
import yaml
import curses
from typing import List, Dict, Any, Set, Union, Tuple
from curses import window

# Static file paths
active_ai_yaml_path: str = "./config/active_ai.yaml"
ai_models_directory: str = "./config/models"

# Define a type alias for more complex model data structures if needed
ModelData = Dict[str, Any]

def load_active_ai_models() -> List[str]:
    f: Any
    with open(active_ai_yaml_path, 'r') as f:
        data: Optional[Dict[str, List[str]]] = yaml.safe_load(f)
    if data and 'active_ai_models' in data and isinstance(data['active_ai_models'], list):
        return data['active_ai_models']
    return []

def save_active_ai_models(active_ai_models: List[str]) -> None:
    f: Any
    with open(active_ai_yaml_path, 'w') as f:
        yaml.dump({'active_ai_models': active_ai_models}, f, default_flow_style=False, sort_keys=False)

def load_available_ai_models() -> List[ModelData]:
    models: List[ModelData] = []
    file: str
    for file in os.listdir(ai_models_directory):
        if file.endswith('.yaml'):
            yaml_path: str = os.path.join(ai_models_directory, file)
            f_yaml: Any
            with open(yaml_path, 'r') as f_yaml:
                data: Optional[ModelData] = yaml.safe_load(f_yaml)
            if data and data.get('type') in ['model', 'openai_vlm_client', 'vlm_model'] and data.get('model_category') is not None:
                data['yaml_file_name'] = file.replace(".yaml", "")
                models.append(data)
    return models

def load_model_data(model_file_name: str) -> ModelData:
    yaml_path: str = os.path.join(ai_models_directory, model_file_name + '.yaml')
    f: Any
    with open(yaml_path, 'r') as f:
        data: Optional[ModelData] = yaml.safe_load(f)
        if data is None:
            return {'yaml_file_name': model_file_name.replace(".yaml", ""), 'model_category':["Unknown"], 'model_version': "N/A", 'model_image_size':"N/A", 'model_info':"Error loading data"}
        data['yaml_file_name'] = model_file_name.replace(".yaml", "")
        return data

def display_model(stdscr: window, y: int, x: int, model: ModelData, highlight: bool = False, incompatible: bool = False, reason: str = "") -> None:
    categories_list: List[str]
    if isinstance(model.get('model_category'), list):
        categories_list = model['model_category']
    elif isinstance(model.get('model_category'), str):
        categories_list = [model['model_category']]
    else:
        categories_list = ["Unknown"]
    categories: str = ", ".join(categories_list)
    
    file_name_display: str = str(model.get('yaml_file_name', "N/A"))
    version_display: str = str(model.get('model_version', "N/A"))
    image_size_display: Any = model.get('model_image_size', "N/A")
    info_display: str = str(model.get('model_info', "N/A"))

    display_text: str = f"{file_name_display:<20} | {categories:<15} | {version_display:<5} | {str(image_size_display):<5} | {info_display:<30}"
    if highlight:
        stdscr.addstr(y, x, display_text, curses.A_REVERSE)
    elif incompatible:
        stdscr.addstr(y, x, display_text + f" (Incompatible: {reason})", curses.color_pair(1))
    else:
        stdscr.addstr(y, x, display_text)

def choose_active_models() -> None:
    curses.wrapper(open_ui)

def open_ui(stdscr: window) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    stdscr.clear()

    active_ai_model_names: List[str] = load_active_ai_models()
    available_ai_models: List[ModelData] = load_available_ai_models()

    active_ai_models: List[ModelData] = [load_model_data(model_name) for model_name in active_ai_model_names]

    available_ai_models = [model for model in available_ai_models if model.get('yaml_file_name') not in active_ai_model_names]

    available_ai_models.sort(key=lambda x: (x.get('model_category',[""])[0] if isinstance(x.get('model_category'),list) and x.get('model_category') else str(x.get('model_category',"")), str(x.get('model_identifier',""))))
    active_ai_models.sort(key=lambda x: (x.get('model_category',[""])[0] if isinstance(x.get('model_category'),list) and x.get('model_category') else str(x.get('model_category',"")), str(x.get('model_identifier',""))))
    
    active_image_sizes: Set[Any] = {model['model_image_size'] for model in active_ai_models if 'model_image_size' in model}
    active_categories: Set[str] = set()
    model_data_item: ModelData
    for model_data_item in active_ai_models:
        category_val: Union[str, List[str], None] = model_data_item.get('model_category')
        if isinstance(category_val, str):
            active_categories.add(category_val)
        elif isinstance(category_val, list):
            cat_item: str
            for cat_item in category_val:
                if isinstance(cat_item, str):
                    active_categories.add(cat_item)

    current_list: str = 'available'
    current_index: int = 0
    idx: int
    model: ModelData

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available AI Models:")
        stdscr.addstr(1, 0, f"{'File Name':<20} | {'Category':<15} | {'Ver':<5} | {'Size':<5} | {'Info':<30}")
        stdscr.addstr(2, 0, "-" * 80)
        y_offset: int = 3
        for idx, model in enumerate(available_ai_models):
            highlight: bool = current_list == 'available' and idx == current_index
            model_img_size: Any = model.get('model_image_size')
            model_cat_list: Union[str, List[str], None] = model.get('model_category')
            
            incompatible_size: bool = any(size != model_img_size for size in active_image_sizes if model_img_size is not None)
            incompatible_cat: bool = False
            current_model_cats: List[str] = []
            if isinstance(model_cat_list, str):
                current_model_cats = [model_cat_list]
            elif isinstance(model_cat_list, list):
                current_model_cats = [str(c) for c in model_cat_list]
            
            cat_check: str
            for cat_check in current_model_cats:
                if cat_check in active_categories:
                    incompatible_cat = True
                    break
            
            incompatible: bool = incompatible_size or incompatible_cat
            reason: str = ""
            if incompatible_size: reason += "Size mismatch. "
            if incompatible_cat: reason += "Category active."
            reason = reason.strip()

            display_model(stdscr, y_offset, 0, model, highlight, incompatible, reason)
            y_offset += 1

        stdscr.addstr(y_offset + 1, 0, "Active AI Models:")
        stdscr.addstr(y_offset + 2, 0, f"{'File Name':<20} | {'Category':<15} | {'Ver':<5} | {'Size':<5} | {'Info':<30}")
        stdscr.addstr(y_offset + 3, 0, "-" * 80)
        y_offset += 4
        for idx, model in enumerate(active_ai_models):
            highlight = current_list == 'active' and idx == current_index
            display_model(stdscr, y_offset, 0, model, highlight)
            y_offset += 1

        stdscr.addstr(y_offset + 2, 0, "Keys: UP/DOWN | ENTER/RIGHT/LEFT - Move | TAB - Switch | q - Quit")

        key: int = stdscr.getch()

        if key == curses.KEY_UP:
            if current_index > 0:
                current_index -= 1
        elif key == curses.KEY_DOWN:
            if current_list == 'available' and current_index < len(available_ai_models) - 1:
                current_index += 1
            elif current_list == 'active' and current_index < len(active_ai_models) - 1:
                current_index += 1
        elif key == curses.KEY_RIGHT or key == curses.KEY_LEFT or key in [curses.KEY_ENTER, 10, 13]:
            move_to_active: bool = (key == curses.KEY_RIGHT or key in [curses.KEY_ENTER, 10, 13])
            move_to_available: bool = key == curses.KEY_LEFT

            if current_list == 'available' and available_ai_models and move_to_active:
                model_to_move: ModelData = available_ai_models[current_index]
                model_img_size = model_to_move.get('model_image_size')
                model_cat_list = model_to_move.get('model_category')
                incompatible_size = any(size != model_img_size for size in active_image_sizes if model_img_size is not None)
                incompatible_cat = False
                current_model_cats = [model_cat_list] if isinstance(model_cat_list, str) else model_cat_list if isinstance(model_cat_list, list) else []
                for cat_check in current_model_cats:
                    if str(cat_check) in active_categories:
                        incompatible_cat = True; break
                if not (incompatible_size or incompatible_cat):
                    active_ai_models.append(available_ai_models.pop(current_index))
                    if model_img_size is not None: active_image_sizes.add(model_img_size)
                    active_categories.update(str(c) for c in current_model_cats)
                    if current_index >= len(available_ai_models): current_index = max(0, len(available_ai_models) - 1)
            elif current_list == 'active' and active_ai_models and move_to_available:
                model_to_move = active_ai_models.pop(current_index)
                available_ai_models.append(load_model_data(model_to_move['yaml_file_name']))
                active_image_sizes = {m['model_image_size'] for m in active_ai_models if 'model_image_size' in m}
                active_categories = {c for m in active_ai_models for c_list in [m.get('model_category', [])] for c in ([c_list] if isinstance(c_list, str) else c_list) if isinstance(c,str)}
                if current_index >= len(active_ai_models): current_index = max(0, len(active_ai_models) - 1)
            
            available_ai_models.sort(key=lambda x: (x.get('model_category',[""])[0] if isinstance(x.get('model_category'),list) and x.get('model_category') else str(x.get('model_category',"")), str(x.get('model_identifier',""))))
            active_ai_models.sort(key=lambda x: (x.get('model_category',[""])[0] if isinstance(x.get('model_category'),list) and x.get('model_category') else str(x.get('model_category',"")), str(x.get('model_identifier',""))))

        elif key == ord('q'):
            break
        elif key == ord('\t'):
            current_list = 'active' if current_list == 'available' else 'available'
            current_index = 0

        stdscr.refresh()

    save_active_ai_models([model['yaml_file_name'] for model in active_ai_models if 'yaml_file_name' in model])

if __name__ == "__main__":
    choose_active_models()
