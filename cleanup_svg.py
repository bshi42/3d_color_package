import xml.etree.ElementTree as ET

tree = ET.parse("aim2_method_v2.svg")
root = tree.getroot()
layer = root[2]

print(f"Original children count: {len(layer)}")

to_keep = []
for child in layer:
    if "id" in child.attrib:
        to_keep.append(child)

print(f"Keeping {len(to_keep)} children with IDs.")

layer.clear()
layer.extend(to_keep)

root.set("viewBox", "0 0 1000 850")
if 'width' in root.attrib:
    del root.attrib['width']
if 'height' in root.attrib:
    del root.attrib['height']

tree.write("aim2_method_v2.svg", encoding="utf-8", xml_declaration=True)
print("Cleanup complete.")
