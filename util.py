def fetch_sub_envoy(envoy, target):
    if envoy._module_path == target:
        return envoy
    
    for sub in envoy._sub_envoys:
        result = fetch_sub_envoy(sub, target)  
        if result is not None:
            return result