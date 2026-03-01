import pandas as pd
from rdkit import Chem
from neo4j import GraphDatabase
from tqdm import tqdm
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Max1432306@#"
NEO4J_DATABASE = "Drug Graph uk"

ZINC_CSV_PATH = "ZINC_250k.csv"
BATCH_SIZE = 500


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


def create_molecule(tx, mol_id, smiles, atoms, bonds):

    tx.run(
        """
        MERGE (m:Molecule {id: $mid})
        SET m.smiles = $smiles
        """,
        mid=mol_id,
        smiles=smiles
    )

    for atom_id, element in atoms.items():
        tx.run(
            """
            MERGE (a:Atom {id: $aid})
            SET a.element = $el
            WITH a
            MATCH (m:Molecule {id: $mid})
            MERGE (m)-[:HAS_ATOM]->(a)
            """,
            aid=f"{mol_id}_{atom_id}",
            el=element,
            mid=mol_id
        )

    for a1, a2, bond_type in bonds:
        tx.run(
            """
            MATCH (a1:Atom {id: $a1}), (a2:Atom {id: $a2})
            MERGE (a1)-[:BONDED_TO {type: $bt}]-(a2)
            """,
            a1=f"{mol_id}_{a1}",
            a2=f"{mol_id}_{a2}",
            bt=bond_type
        )

def load_zinc():

    print("Loading ZINC CSV...")
    df = pd.read_csv(ZINC_CSV_PATH)

    smiles_col = None
    for col in df.columns:
        if col.lower() in ["smiles", "canonical_smiles"]:
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError("No SMILES column found in CSV")

    smiles_list = df[smiles_col].dropna().tolist()

    print(f"Found {len(smiles_list)} SMILES strings")

    with driver.session(database=NEO4J_DATABASE) as session:

        batch = []

        for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            mol_id = f"ZINC_{idx}"

            atoms = {
                atom.GetIdx(): atom.GetSymbol()
                for atom in mol.GetAtoms()
            }

            bonds = []
            for bond in mol.GetBonds():
                bonds.append((
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    str(bond.GetBondType())
                ))

            batch.append((mol_id, smiles, atoms, bonds))

            if len(batch) >= BATCH_SIZE:
                session.execute_write(
                    lambda tx: [
                        create_molecule(tx, *entry) for entry in batch
                    ]
                )
                batch.clear()

        if batch:
            session.execute_write(
                lambda tx: [
                    create_molecule(tx, *entry) for entry in batch
                ]
            )

    print("ZINC loading completed successfully")


if __name__ == "__main__":
    load_zinc()
