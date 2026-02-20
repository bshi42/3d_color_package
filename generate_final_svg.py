import xml.etree.ElementTree as ET

# Read the original SVG
ET.register_namespace("", "http://www.w3.org/2000/svg")
tree = ET.parse('aim2_method_v2.svg')
root = tree.getroot()
ns = {'svg': 'http://www.w3.org/2000/svg'}

print("Original viewBox:", root.attrib.get('viewBox'))
print("Number of children in root:", len(root))
for child in root:
    print(child.tag, child.attrib.get('id', ''))
