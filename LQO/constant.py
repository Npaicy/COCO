# Hint Type
LEADINGHINT = 0
RULEHINT = 1
# Postgres Data Type
PGDATATYPE = ['smallint','integer','bigint','decimal','numeric','real',
                'double precision','smallserial','serial','bigserial']
PGCHARTYPE = ['character','character varying','text','bytea']
PGDATETYPE = ['date','timestamp','interval','timestamp without time zone','time without time zone']

JOINTYPE = ["Nested Loop", "Hash Join", "Merge Join"]
SCANTYPE = ['Index Only Scan', 'Seq Scan', 'Index Scan', 'Bitmap Heap Scan','Tid Scan']
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Recheck Cond','Filter']
BINOP = [' >= ',' <= ',' = ',' > ',' < ']
OP2IDX ={'=ANY': 0,'>=':1,'<=':2,'>': 3,'=': 4,'<': 5,'NA':6,'IS NULL':7,'IS NOT NULL':8, '<>':9,'~~':10,'!~~':11, '~~*': 12,'<>ALL':13}
TYPE2IDX = {
            "Aggregate": 1,
            "Nested Loop": 2,
            "Seq Scan": 3,
            "Index Scan": 4,
            "Hash Join": 5,
            "Hash": 6,
            "Merge Join": 7,
            "Sort": 8,
            "Gather": 9,
            "Materialize": 10,
            "Index Only Scan": 11,
            "Bitmap Heap Scan": 12,
            "Bitmap Index Scan": 13,
            "Gather Merge": 14,
            "Limit": 15,
            "WindowAgg": 16,
            "Group": 17,
            'Unique': 18,
            'Incremental Sort': 19,
            'BitmapOr':20,
            'Result':21,
            'BitmapAnd':22
        }

ALLRULES = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"]

OPERATOR2HINT = {'Index Only Scan':'IndexOnlyScan', 'Seq Scan':'SeqScan', 
                     'Index Scan':'IndexScan', 'Bitmap Heap Scan':'BitmapScan','Tid Scan':'TidScan',
                     'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}

ACTION2HINT = {
            0:'SET enable_nestloop TO off;',
            1:'SET enable_hashjoin TO off;',
            2:'SET enable_mergejoin TO off;',
            3:'SET enable_seqscan TO off;',
            4:'SET enable_indexscan TO off;',
            5:'SET enable_indexonlyscan TO off;',
            6:'SET enable_bitmapscan TO off;'}
HINT2POS = {"NESTLOOP":0, "HASHJOIN":1, "MERGEJOIN":2, "SeqScan":3, "IndexScan":4, "IndexOnlyScan":5, 'BitmapScan':6}

PLANMAXTIMEOUT = 3e5
MAXNODE = 60
HEIGHTSIZE = 35
MAXFILTER = 15
MAXJOIN = 10
MAXDISTANCE = 20