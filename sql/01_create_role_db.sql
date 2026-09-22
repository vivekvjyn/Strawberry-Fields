-- Run once as the postgres superuser:
--   sudo -u postgres psql -f sql/01_create_role_db.sql
--
-- Creates the login role and database that .env points at
-- (DB_USER=vivek, DB_NAME=strawberry_fields).

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'vivek') THEN
        CREATE ROLE vivek WITH LOGIN PASSWORD '1341';
    END IF;
END
$$;

SELECT 'CREATE DATABASE strawberry_fields OWNER vivek'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'strawberry_fields')
\gexec
