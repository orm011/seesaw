import inspect

def copy_locals():
    '''
        copies all local variables from this context into the jupyter top level, eg, for easier 
        debugging of data and for prototyping new code that is eventually meant to run within this context.
    '''
    stack = inspect.stack()
    
    caller = stack[1]
    local_dict = {k:v  for k,v in caller.frame.f_locals.items() if not k.startswith('_')}
    
    notebook_caller = None
    for st in stack:
        if st.function == '<module>':        
            notebook_caller = st
            break
            
    if notebook_caller is None:
        print('is this being called from within a jupyter notebook?')
        return
    
    print('copying variables to <module> globals...', list(local_dict.keys()))
    notebook_caller.frame.f_globals.update(local_dict)