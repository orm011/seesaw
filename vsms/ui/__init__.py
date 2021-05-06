import os
import ipyvue as vue

uipath = os.path.dirname(__file__)
component_files =  [fn for fn in os.listdir(uipath) if fn.endswith('.vue')]

## re-register components on vue upon reload to reflect changes
for filename in component_files:
    compname = filename.split('.')[0]
    definition = open(f'{uipath}/{filename}').read()
    if definition.find('export default') < 0:
        print('did not find export default in module. check spacing etc.')
    definition = definition.replace('export default', 'module.exports = ')
    vue.register_component_from_string(compname, definition)
