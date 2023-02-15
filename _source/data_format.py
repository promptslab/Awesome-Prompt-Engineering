def convert_data(data):
    categories = {}
    for item in data:
        if item['cc'] not in categories:
            categories[item['cc']] = []
        categories[item['cc']].append(f"[{item['p']}]({item['i']}) [{item['y']}]")
    
    result = ""
    for cat, items in categories.items():
        result += f"- {cat}:\n"
        for item in items:
            result += f"  - {item}\n"
    
    return result
