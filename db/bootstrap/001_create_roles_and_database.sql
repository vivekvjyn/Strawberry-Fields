\set ON_ERROR_STOP on

SELECT format('CREATE ROLE %I WITH LOGIN PASSWORD %L', :'app_role', :'app_password')
WHERE NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles WHERE rolname = :'app_role'
)
\gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'database_name', :'app_role')
WHERE NOT EXISTS (
    SELECT FROM pg_catalog.pg_database WHERE datname = :'database_name'
)
\gexec
