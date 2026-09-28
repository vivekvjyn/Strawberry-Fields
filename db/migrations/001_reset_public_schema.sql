DO $$
DECLARE
    relation_name TEXT;
BEGIN
    FOR relation_name IN
        SELECT viewname FROM pg_catalog.pg_views WHERE schemaname = 'public'
    LOOP
        EXECUTE format('DROP VIEW IF EXISTS public.%I CASCADE', relation_name);
    END LOOP;

    FOR relation_name IN
        SELECT tablename FROM pg_catalog.pg_tables
        WHERE schemaname = 'public' AND tablename <> 'schema_migrations'
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS public.%I CASCADE', relation_name);
    END LOOP;
END
$$;
