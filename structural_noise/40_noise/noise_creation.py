import pandas as pd
from pykeen.triples import TriplesFactory
import torch
from collections import defaultdict
import random

# Load original dataset
df_original = pd.read_csv('positive_dataset.csv')

# Extract relevant columns
df_original = df_original[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI', 'SUBJECT_SEMTYPE', 'OBJECT_SEMTYPE']]

# Prepare training triples
triples_factory = TriplesFactory.from_labeled_triples(triples=df_original[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']].values)

# Extract unique entities and map to their types (using names, not integers)
entities = pd.concat([df_original['SUBJECT_CUI'], df_original['OBJECT_CUI']]).unique()
original_entity_set = set(entities)

# Map entities to their semantic types (CUI → Type Name)
entity_types_map = {}  # Maps entity CUI to its type name
unique_semtypes = set()

for _, row in df_original.iterrows():
    subject_cui, object_cui = row['SUBJECT_CUI'], row['OBJECT_CUI']
    
    # Extract the first semantic type (if multiple exist)
    subject_semtype = row['SUBJECT_SEMTYPE'].split(',')[0]  
    object_semtype = row['OBJECT_SEMTYPE'].split(',')[0]
    
    unique_semtypes.update([subject_semtype, object_semtype])  # Track unique types
    
    # Map CUIs to semantic type names
    entity_types_map[subject_cui] = subject_semtype
    entity_types_map[object_cui] = object_semtype

# Create a **mapping from semantic type IDs to type names**
semtype_to_int = {semtype: idx for idx, semtype in enumerate(unique_semtypes)}
int_to_semtype = {idx: semtype for semtype, idx in semtype_to_int.items()}  # Reverse mapping

# Assign entity IDs to **semantic type names** instead of numbers
entity_types = {}  # Maps entity ID to semantic type name
entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

for entity, entity_id in entity_to_id.items():
    # Retrieve entity type **name**, not integer ID
    entity_types[entity_id] = entity_types_map.get(entity, "Unknown")  # Use "Unknown" if missing


# Reverse mappings
id_to_entity = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}



import torch
import random
from collections import defaultdict
from collections import defaultdict
import torch
import random

class NoiseGenerator:
    def __init__(self, triples_factory, entity_types, corruption_ratio=0.4, max_attempts=100):
        self.triples_factory = triples_factory
        self.entity_types = entity_types
        self.corruption_ratio = corruption_ratio
        self.max_attempts = max_attempts
        self.positive_triples = self.triples_factory.mapped_triples
        self.relation_to_id = self.triples_factory.relation_to_id
        self.whole_set = {tuple(triple.cpu().numpy()) for triple in self.positive_triples}
        self.relation_counts = defaultdict(int)
        self.entity_counts = defaultdict(int)
        self.corrupted_dataset = self.positive_triples.clone()  # Initialize corrupted dataset

        for triple in self.positive_triples:
            for entity in (triple[0].item(), triple[2].item()):
                self.entity_counts[entity] += 1
            self.relation_counts[triple[1].item()] += 1

    def generate_noise(self):
        unique_negatives = set()
        corrupted_triples = []
        failed_corruptions = 0  # Counter for failed corruption attempts

        # Calculate corruption count for each relation based on 10% of dataset
        total_to_corrupt = int(len(self.positive_triples) * self.corruption_ratio)
        relation_corruption_counts = {
            rel_id: int((count / len(self.positive_triples)) * total_to_corrupt)
            for rel_id, count in self.relation_counts.items()
        }

        corrupted_samples = 0
        for relation_id, num_to_corrupt in relation_corruption_counts.items():
            relation_triples = [triple for triple in self.positive_triples if triple[1].item() == relation_id]
            random.shuffle(relation_triples)  # Shuffle to ensure randomness
            selected_triples = relation_triples[:num_to_corrupt]  # Randomly select triples
            
            for positive_triple in selected_triples:
                if corrupted_samples >= total_to_corrupt:
                    break
                
                attempts = 0
                valid_corruption = False

                while not valid_corruption and attempts < self.max_attempts:
                    attempts += 1
                    corrupt_head = torch.rand(1).item() < 0.5
                    candidate_triple = positive_triple.clone()

                    entity_idx = 0 if corrupt_head else 2
                    original_entity = candidate_triple[entity_idx].item()
                    entity_type = self.entity_types.get(original_entity, None)

                    if entity_type is not None:
                        same_type_entities = [
                            entity for entity, etype in self.entity_types.items() if etype == entity_type
                        ]
                        if same_type_entities:
                            new_entity = torch.tensor(same_type_entities)[torch.randint(len(same_type_entities), (1,))]
                            if self.entity_counts[original_entity] > 2:  # Ensure entity is not lost # 1 can be cahnged with 2 
                                candidate_triple[entity_idx] = new_entity.to(self.positive_triples.device)
                                
                                # Update entity counts
                                self.entity_counts[original_entity] -= 1
                                self.entity_counts[new_entity.item()] += 1

                    # Check if corrupted triple is valid
                    candidate_tuple = tuple(candidate_triple.cpu().numpy())
                    if candidate_tuple not in self.whole_set and candidate_tuple not in unique_negatives:
                        unique_negatives.add(candidate_tuple)
                        valid_corruption = True
                        corrupted_triples.append((positive_triple, candidate_triple))  # Store both original and corrupted
                        corrupted_samples += 1
                        
                        # Update the corrupted dataset
                        index = torch.where((self.corrupted_dataset == positive_triple).all(dim=1))[0]
                        if len(index) > 0:
                            self.corrupted_dataset[index[0]] = candidate_triple

                # If all attempts failed, increment the counter
                if not valid_corruption:
                    failed_corruptions += 1
        
        return corrupted_triples, self.relation_counts, self.corrupted_dataset, failed_corruptions






# Generate noise
noise_generator = NoiseGenerator(triples_factory, entity_types)
corrupted_triples, relation_counts, df_corrupted, failed_corruptions = noise_generator.generate_noise()
print("Number of failed corruptions: ",  failed_corruptions)

# Compute the percentage of change per relation type
corruption_percentages = []
for relation_id, count in relation_counts.items():
    relation_name = id_to_relation.get(relation_id, 'Unknown')
    corrupted_count = sum(1 for _, triple in corrupted_triples if triple[1].item() == relation_id)
    percentage = (corrupted_count / count) * 100 if count > 0 else 0
    corruption_percentages.append([relation_name, relation_id, count, corrupted_count, percentage])

df_corruption_info = pd.DataFrame(corruption_percentages, columns=['Relation', 'Relation_ID', 'Original Count', 'Corrupted Count', 'Corruption %'])

print(df_corruption_info)

comparison_data = []
for original, corrupted in corrupted_triples:
    original_subject = id_to_entity[original[0].item()]
    original_object = id_to_entity[original[2].item()]
    corrupted_subject = id_to_entity[corrupted[0].item()]
    corrupted_object = id_to_entity[corrupted[2].item()]
    
    # Fetch entity type **names** from entity_types
    original_subject_type = entity_types.get(original[0].item(), "Unknown")
    original_object_type = entity_types.get(original[2].item(), "Unknown")
    corrupted_subject_type = entity_types.get(corrupted[0].item(), "Unknown")
    corrupted_object_type = entity_types.get(corrupted[2].item(), "Unknown")

    comparison_data.append([
        original_subject,   # Original Subject & Type
        id_to_relation[original[1].item()],       # Relation
        original_object,    # Original Object & Type
        corrupted_subject,  # Corrupted Subject & Type
        corrupted_object,    # Corrupted Object & Type
        original_subject_type,
        corrupted_subject_type,
        original_object_type,
        corrupted_object_type
    ])

# Convert to DataFrame
df_comparison = pd.DataFrame(comparison_data, columns=[
    'Original_Subject', 
    'Relation', 
    'Original_Object',  
    'Corrupted_Subject',  
    'Corrupted_Object', 
    'Original_Subject_Type',
    'Corrupted_Subject_Type',
    'Original_Object_Type',
    'Corrupted_Object_Type'
])

# Save comparison results
df_comparison.to_csv('comparison_dataset.csv', index=False)

print(f"Comparison dataset saved with {len(df_comparison)} changes.")


corrupted_mapped = [
    (id_to_entity[triple[0].item()], id_to_relation[triple[1].item()], id_to_entity[triple[2].item()])
    for triple in df_corrupted
]

df_corrupted = pd.DataFrame(corrupted_mapped, columns=["SUBJECT_CUI", "PREDICATE", "OBJECT_CUI"])


entities = pd.concat([df_corrupted['SUBJECT_CUI'], df_corrupted['OBJECT_CUI']]).unique()
noised_entity_set = set(entities)

df_corrupted.to_csv('noised_dataset.csv', index=False)
print(f"Noised dataset saved with {len(corrupted_triples)} corrupted triples.")


print("Original entity size: ",len(original_entity_set))
print("Noised entity size: ", len(noised_entity_set))

if(len(original_entity_set) != len(noised_entity_set)):
    print("Entity size mismatch!!! Run the code again until you dont see that warning!")
