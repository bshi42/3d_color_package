import re
with open('/Users/leyangloh/dev/3d_color_package/aim2_method_v2_editable.svg', 'r') as f:
    content = f.read()

parts = re.findall(r'<g id="Part_[A-C]_.*?</g></g>', content, flags=re.DOTALL)
if not parts:
    print("Could not parse parts with regex, trying simpler...")
    for part in ['Part_A_Texture_Color', 'Part_B_Multi_Recolor', 'Part_C_Segmentation']:
        idx = content.find(part)
        print(f"{part} found at index {idx}")
else:
    for p in parts:
        print(p[:200] + " ... " + p[-50:])

