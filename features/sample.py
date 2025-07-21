import psycopg2
import json
import os
from typing import Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GenConfig.gen_config import GenConfig

def generate(db_config: Dict[str, str], output_dir: str) -> None:
    databases = GenConfig.databases
    for db_name in databases:
        result = {}
        if db_name in ['template0', 'template1' ,'zhongkai']:
            continue
        current_db_config = db_config.copy()
        current_db_config['database'] = db_name
        
        conn = psycopg2.connect(**current_db_config)
        cursor = conn.cursor()
        print(db_name)
        try:
            cursor.execute(f"""
                SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
            """)
            tables = [name[0] for name in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                """)
                columns = cursor.fetchall()
                if table not in result:
                    result[table] = {}
                for column_name, data_type in columns:
                    cursor.execute(f"""
                        SELECT DISTINCT ON (random()) "{column_name}"
                        FROM "{table}"
                        WHERE "{column_name}" IS NOT NULL
                        ORDER BY random()
                        LIMIT 5
                    """)
                    samples = [str(row[0]) for row in cursor.fetchall()]
                    if column_name not in result[table]:
                        result[table][column_name]= {
                            "data_type": data_type,
                            "samples": samples
                    }
            with open(os.path.join(output_dir,db_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    db_config = GenConfig.db_config
    
    generate(db_config, GenConfig.sample_dir)