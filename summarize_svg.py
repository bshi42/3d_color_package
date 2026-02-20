import xml.etree.ElementTree as ET

def print_tree(elem, level=0, max_level=3):
    if level > max_level: return
    tag = elem.tag.split('}')[-1]
    _id = elem.attrib.get('id', '')
    label = elem.attrib.get('{http://www.inkscape.org/namespaces/inkscape}label', '')
    text_content = ""
    if tag in ['text', 'tspan']:
        text_content = " -> " + (elem.text or "").strip()
    print("  " * level + f"<{tag} id='{_id}' label='{label}'>{text_content}")
    for child in elem:
        print_tree(child, level + 1, max_level)

tree = ET.parse('/Users/leyangloh/dev/3d_color_package/aim2_method_v2_editable.svg')
print_tree(tree.getroot(), max_level=3)
