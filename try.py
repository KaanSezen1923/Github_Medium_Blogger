import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# .env dosyanızdan şifrenizi çekiyoruz
password = os.getenv("SUPEBASE_PASSWORD")

# Supabase'den aldığınız bilgilere göre oluşturulan bağlantı dizesi
conn_str = f"postgresql://postgres.yspgeqqsqrzirlvjessn:{password}@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"

try:
    print("Supabase veritabanına bağlanılıyor...")
    connection = psycopg2.connect(conn_str)
    cursor = connection.cursor()
    
    # Basit bir test sorgusu
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    
    print("\n[BAŞARILI] Supabase (PostgreSQL) bağlantısı başarıyla kuruldu!")
    print(f"Veritabanı Sürümü: {db_version[0]}\n")
    
    cursor.close()
    connection.close()
    
except Exception as e:
    print("\n[HATA] Bağlantı kurulamadı:")
    print(e)