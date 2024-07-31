import pandas as pd
import random
from datetime import datetime, timedelta

def generate_id(prefix, index):
    return f"{prefix}{index:03d}"

def generate_timestamp():
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2021, 12, 31)
    return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

def generate_cost(base_cost):
    return max(100, int(base_cost * random.uniform(0.8, 1.2)))

def generate_lead_time(base_time):
    return max(1, int(base_time * random.uniform(0.9, 1.1)))

def generate_inventory(base_inventory):
    return max(0, int(base_inventory * random.uniform(0.7, 1.3)))

item_types = {
    'final_product': {'prefix': 'FP', 'cost_range': (10000000, 20000000), 'lead_time_range': (150, 210), 'inventory_range': (2, 5)},
    'system': {'prefix': 'SY', 'cost_range': (2000000, 8000000), 'lead_time_range': (60, 120), 'inventory_range': (3, 8)},
    'subsystem': {'prefix': 'SS', 'cost_range': (500000, 2000000), 'lead_time_range': (30, 90), 'inventory_range': (5, 15)},
    'component': {'prefix': 'CP', 'cost_range': (50000, 500000), 'lead_time_range': (15, 45), 'inventory_range': (10, 30)},
    'subcomponent': {'prefix': 'SC', 'cost_range': (5000, 50000), 'lead_time_range': (5, 20), 'inventory_range': (20, 50)},
    'raw_material': {'prefix': 'RM', 'cost_range': (500, 5000), 'lead_time_range': (1, 10), 'inventory_range': (50, 100)},
}

suppliers = [f"Supplier_{i:03d}" for i in range(1, 45)]
distributors = [f"Distributor_{i:03d}" for i in range(1, 15)]
transport_methods = ['Truck', 'Air Freight', 'Sea Freight', 'Rail']

def generate_item_pool(num_items_per_type):
    item_pool = {}
    for item_type, type_info in item_types.items():
        item_pool[item_type] = []
        for i in range(num_items_per_type):
            item_id = generate_id(type_info['prefix'], i + 1)
            item_pool[item_type].append({
                'item_id': item_id,
                'item_name': f"{item_type.capitalize()} {item_id}",
                'item_type': item_type,
            })
    return item_pool

def generate_item(item_type, parent_id, item_pool):
    item = random.choice(item_pool[item_type]).copy()
    type_info = item_types[item_type]
    
    item.update({
        'parent_id': parent_id,
        'quantity': random.randint(1, 5),
        'lead_time': generate_lead_time(random.randint(*type_info['lead_time_range'])),
        'cost': generate_cost(random.randint(*type_info['cost_range'])),
        'supplier': random.choice(suppliers),
        'inventory': generate_inventory(random.randint(*type_info['inventory_range'])),
        'min_stock': random.randint(1, 5),
        'assembly_time': random.randint(0, 72),
        'distributor': random.choice(distributors),
        'transport_method': random.choice(transport_methods),
        'transport_cost': random.randint(100, 10000),
        'transport_time': random.randint(1, 30),
    })
    return item

def generate_complex_dataset(num_records):
    data = []
    item_hierarchy = {}
    item_pool = generate_item_pool(20)  # Create a pool of 20 items for each type

    # Generate final products
    num_final_products = random.randint(3, 5)
    dummy_parent_id = 'DUMMY_PARENT'
    for _ in range(num_final_products):
        final_product = generate_item('final_product', dummy_parent_id, item_pool)
        data.append(final_product)
        item_hierarchy[final_product['item_id']] = {'systems': []}

    # Generate systems, subsystems, components, subcomponents, and raw materials
    while len(data) < num_records:
        for fp_id, fp_info in item_hierarchy.items():
            # Add systems
            if len(fp_info['systems']) < random.randint(3, 5):
                system = generate_item('system', fp_id, item_pool)
                data.append(system)
                fp_info['systems'].append({'id': system['item_id'], 'subsystems': []})

            # Add subsystems, components, subcomponents, and raw materials
            for system in fp_info['systems']:
                if len(system['subsystems']) < random.randint(3, 7):
                    subsystem = generate_item('subsystem', system['id'], item_pool)
                    data.append(subsystem)
                    system['subsystems'].append({'id': subsystem['item_id'], 'components': []})

                for subsystem in system['subsystems']:
                    if len(subsystem['components']) < random.randint(5, 10):
                        component = generate_item('component', subsystem['id'], item_pool)
                        data.append(component)
                        subsystem['components'].append({'id': component['item_id'], 'subcomponents': []})

                    for component in subsystem['components']:
                        if len(component['subcomponents']) < random.randint(2, 5):
                            subcomponent = generate_item('subcomponent', component['id'], item_pool)
                            data.append(subcomponent)
                            component['subcomponents'].append(subcomponent['item_id'])

                        # Add raw materials
                        if random.random() < 0.3:  # 30% chance to add raw material
                            raw_material = generate_item('raw_material', component['id'], item_pool)
                            data.append(raw_material)

        if len(data) >= num_records:
            break

    # Add timestamps
    for item in data:
        item['timestamp'] = generate_timestamp()

    return pd.DataFrame(data)

# Generate the complex dataset
num_records = 1000
df = generate_complex_dataset(num_records)

# Save to CSV
df.to_csv('complex_lithography_data.csv', index=False)
print(f"Complex dataset with {len(df)} records saved to 'complex_lithography_data.csv'")

# Print some statistics
print("\nDataset Statistics:")
print(f"Total records: {len(df)}")
print(f"Unique item types: {df['item_type'].nunique()}")
print("\nRecords per item type:")
print(df['item_type'].value_counts())
print("\nUnique items per type:")
for item_type in df['item_type'].unique():
    print(f"{item_type}: {df[df['item_type'] == item_type]['item_id'].nunique()}")
print("\nUnique suppliers:", df['supplier'].nunique())
print("Unique distributors:", df['distributor'].nunique())
print("\nDate range:")
print(f"Start: {df['timestamp'].min()}")
print(f"End: {df['timestamp'].max()}")
