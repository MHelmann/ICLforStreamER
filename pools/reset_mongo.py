import sys
sys.path.append('../')

from pymongo import MongoClient
from meta_data.metadata import MetaData 


def reset():

        host = "localhost"
        port = 27017
        try:
            client = MongoClient(
                host=host,
                port=port,
                username="Helmann",
                password="helmanmm"

            )
        except Exception:
            raise
        else:
            label_db_name = "label_pool"
            primary_coln_name = "label_primary"
            enriched_coln_name = "label_secondary"

            label_pool_db = client[label_db_name]
            primary_coln = label_pool_db[primary_coln_name]
            enriched_coln = label_pool_db[enriched_coln_name]

            primary_coln.drop()
            enriched_coln.drop()

            cand_db_name = "candidate_pool"
            second_coln_name = "cand_secondary"
            cand_pool_db = client[cand_db_name]
            second_coln = cand_pool_db[second_coln_name]

            second_coln.drop()

            flag_db_name = "flags"
            flag_coln_name = "flag_col"
            flags_db = client[flag_db_name]
            flag_coln = flags_db[flag_coln_name]

            flag_coln.update_one({'_id':1}, {"$set":{'flag': False}} )

            cursor = flag_coln.find()
            for record in cursor: 
                print(record)

            databases = client.list_database_names()

            # Iterate over databases
            for db_name in databases:

                if db_name in ["local", "config"]:
                    print(db_name)
                    continue
                else:
                    print(f"Database: {db_name}")
                    db = client[db_name]

                    # List collections for each database
                    collections = db.list_collection_names()
                    for collection_name in collections:
                        print(f"\tCollection: {collection_name}")

            # Close connection
            client.close()
            

            m_data = MetaData()
            meta_data = m_data.get_meta_data()
            timestamp = 1710085166.703743
            meta_data["pool"]["last_store_ts"] = timestamp
            meta_data["label_pool"]["last_fetch_ts"] = timestamp
            meta_data["icl_pool"]["last_fetch_ts"] = timestamp
            m_data.set_meta_data(meta_data)
        
if __name__ == '__main__':
    reset()
