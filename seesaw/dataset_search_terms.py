def clean_space(x):
    x = x.strip()
    cx = " ".join([w for w in x.split()])
    return cx


def clean_objectnet_name(name):
    ## bills (money) => money bills
    ## binder (closed) => closed binder
    ## coffee/french press => coffee or french press
    elts = name.split("(")  ## assert only two parts
    if len(elts) == 2:
        noun, adj = elts
        adj = adj.replace(")", " ")
        name = " ".join([adj, noun])

    name = name.replace("/", " or ")
    return clean_space(name)


def clean_lvis_name(name):
    return clean_space(name.replace("_", " ").replace("(", " ").replace(")", " "))


def clean_dota_name(name):
    return name.replace("-", " ")


_clean_function = {
    "objectnet": clean_objectnet_name,
    "lvis": clean_lvis_name,
    "dota": clean_dota_name,
    'lvispatch' : clean_lvis_name,
}

_special_cases = {
    "bdd": {
        "motor": "motorcycle",
        "rider": "bike rider",
        "gas stations scene": "gas station",
        "trailer": "trailer hitched to a car",
        "highway scene": "highway",
        "parking lot scene": "parking lot",
        "city street scene": "city street",
        "residential scene": "residential street",
        "tunnel scene": "tunnel",
        "overcast weather": "overcast sky",
        "partly cloudy weather": "partly cloudy sky",
        "clear weather": "clear skies",
        "foggy weather": "foggy weather",
        "wheelchair" : "wheelchair",
    },
    "coco": {"mouse": "computer mouse"},
}


def category2query(dataset, cat):
    if cat in _special_cases.get(dataset, {}):
        return _special_cases[dataset][cat]
    else:
        clnfn = _clean_function.get(dataset, lambda x: x)
        return clnfn(cat)
