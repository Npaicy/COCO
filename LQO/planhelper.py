import pickle
import numpy as np
from collections import deque,defaultdict
import math
from LQO.util import swap_dict_items, minmax_transform
from LQO.pghelper import PGHelper
from LQO.constant import *
from LQO.util_plan import *

class PlanHelper:
    def __init__(self, config, build_pghelper = False):
        if build_pghelper:
            self.pghelper = PGHelper(config.dbConfig)
        else:
            self.pghelper = None
        self.load_meta_info(config.meta_info_path, config.assindex_path)
        self.alias2table = {dbnames: {} for dbnames in config.databases}
        self.get_alias2table()
        # self.transform_statistics = {'Plan Rows':(0.6931,26.6003), 'Total Cost':(0.0,27.8201), 'Plan Width':(0.0,5.2627), 'Startup Cost':(0.0,27.8201)}

    def get_alias2table(self):
        self.alias2table['imdb'] = {'chn': 'char_name', 'ci': 'cast_info', 'cn': 'company_name', 
                                    'ct': 'company_type', 'mc': 'movie_companies', 'rt': 'role_type', 'c':'char_name',
                                    't': 'title', 'k': 'keyword', 'lt': 'link_type', 'mk': 'movie_keyword', 
                                    'ml': 'movie_link', 'it1': 'info_type', 'it2': 'info_type', 'mi': 'movie_info', 
                                    'mi_idx': 'movie_info_idx', 'it': 'info_type', 'kt': 'kind_type', 'miidx': 'movie_info_idx', 
                                    'aka_t': 'aka_title', 'at': 'aka_title', 'an': 'aka_name', 'n': 'name', 'cc': 'complete_cast', 
                                    'cct1': 'comp_cast_type', 'cct2': 'comp_cast_type', 'it3': 'info_type', 'pi': 'person_info', 
                                    't1': 'title', 't2': 'title', 'cn1': 'company_name', 'cn2': 'company_name', 'kt1': 'kind_type',
                                    'kt2': 'kind_type', 'mc1': 'movie_companies', 'mc2': 'movie_companies', 'mi_idx1': 'movie_info_idx', 
                                    'mi_idx2': 'movie_info_idx', 'an1': 'aka_name', 'n1': 'name', 'a1': 'aka_name'}
        self.alias2table['stats'] = {'st_b':'badges','st_c':'comments','st_p':'posts','st_u':'users','st_v':'votes','st_t':'tags','st_pl':'postlinks','st_ph':'posthistory'}
        for dbname in self.alias2table.keys():
            for table_name in self.metaInfo[dbname]['tableAttr'].keys():
                self.alias2table[dbname][table_name] = table_name

    def load_meta_info(self, metaInfoPath, assindexPath):
        self.metaInfo = pickle.load(open(metaInfoPath, 'rb'))
        self.assindex = pickle.load(open(assindexPath, 'rb'))

    def transform_hint(self, hintcode):
        binary_indices = [j for j in range(hintcode.bit_length()) if (hintcode >> j) & 1]
        hints = []
        for j in binary_indices:
            hints.append(ACTION2HINT[j])
        return hints

    def get_latency_analyze(self, hint, sql, dbName, timeout = PLANMAXTIMEOUT, hintstyle = RULEHINT): 
        if hintstyle == RULEHINT:
            if isinstance(hint, str):
                hint = self.transform_hint(int(hint))
            elif isinstance(hint, int):
                hint = self.transform_hint(hint)
        timeout = min(PLANMAXTIMEOUT, timeout)
        timeout = max(timeout, 100)
        latency, istimeout, planJson = self.pghelper.get_latency_plan(hint, sql, dbName, timeout, hintstyle)
        return latency, istimeout, planJson
    
    def test_sql_meaningful(self,sql,dbName):
        return self.pghelper.get_result(sql,dbName)
    
    def test_sql_validity(self,sql,dbName):
        return self.get_feature(0, sql,dbName=dbName, hintStyle=RULEHINT)
    
    def get_feature(self, hint, query, dbName, hintStyle = RULEHINT):
        if hintStyle == RULEHINT:
            if isinstance(hint, str):
                hint = self.transform_hint(int(hint))
            elif isinstance(hint, int):
                hint = self.transform_hint(hint)
        planJson = self.pghelper.get_cost_plan(hint, query, hintStyle, dbName = dbName)
        try:
            featureDict, hintDict, left_deep = self.get_feature_from_planJson(planJson, dbName)
            return featureDict, hintDict, left_deep, planJson
        except:
            print(dbName, planJson)
            return None
        
    
    def get_feature_from_planJson(self, planJson, dbName, executed = False):
        hintDict = {"scan table": {}, "join operator": {}, "scan operator": {}}
        dbName = dbName.split('_')[0]
        try:
            featureDict = self.traverse_plan(planJson["Plan"], hintDict, dbName, executed=executed)
        except:
            print(dbName, planJson)
            raise ValueError('Plan Parse Error')
        featureDict = self.pre_collate(featureDict, executed=executed)
        hintDict, left_deep = self.process_hint(hintDict)
        return featureDict, hintDict, left_deep
    
    def process_hint(self, hint):
        left_deep = True
        ICP = {'join order':[],'join operator':[],'scan operator':[],'structure':[]}
        if len(hint['join operator']) < 1:
            return ICP, None
        hint['join operator'] = dict(reversed(list(hint['join operator'].items())))
        encodOfJoin = list(hint['join operator'].keys())
        for k in range(0,len(encodOfJoin) - 1):
            if len(encodOfJoin[k]) == len(encodOfJoin[k + 1]) and encodOfJoin[k][-1] > encodOfJoin[k + 1][-1]:
                hint['join operator'] = swap_dict_items(hint['join operator'], encodOfJoin[k], encodOfJoin[k + 1])
        padLen = max([len(k) for k in hint['scan table'].keys()])
        sortbyencod = []
        for k in hint['scan table'].keys():
            sum_e = 2 ** (padLen - len(k)) - 1
            for i_e, e in enumerate(k[-1::-1]):
                sum_e += eval(e) * (2 ** (padLen - len(k) + i_e))
            sortbyencod.append((sum_e, k, hint['scan table'][k],hint['scan operator'][k]))
        sortbyencod.sort(key = lambda x: x[0])
        encod = [x[1] for x in sortbyencod]
        jointable = [x[2] for x in sortbyencod]
        ICP['join order'] = [table[0] for _, _, table,_ in sortbyencod]
        ICP['scan operator'] = [OPERATOR2HINT[scan] for _, _, _,scan in sortbyencod]
        for joinEncod in hint['join operator']:
            prefixLen = len(joinEncod)
            JoinE = []
            JoinI = []
            for i_, scanEncod in enumerate(encod):
                if scanEncod[0:prefixLen] == joinEncod:
                    JoinI.append(i_)
                    JoinE.append(scanEncod)
            if len(JoinI) != 2 or (JoinI[1] - JoinI[0]) != 1:
                raise KeyError('Parse Error')
            if JoinI[0] != 0:
                left_deep = False
            encod[JoinI[0]] = joinEncod
            del  encod[JoinI[1]]
            jointable[JoinI[0]] = jointable[JoinI[0]] + jointable[JoinI[1]]
            del jointable[JoinI[1]]
            ICP['structure'].append(JoinI[0])
            ICP['join operator'].append(OPERATOR2HINT[hint['join operator'][joinEncod]])
        return ICP, left_deep
    
    # def extract_sql_info(self, sql, dbname):
    #     parsed_sql = sqlglot.parse_one(sql)
    #     for table in parsed_sql.find_all(sqlglot.exp.Table):
    #         if table.alias_or_name not in self.alias2table[dbname]:
    #             self.alias2table[dbname][table.alias_or_name] = table.name

    def pre_collate(self, theDict, executed):
        x = pad_2d_unsqueeze(theDict['features'], MAXNODE)
        N = len(theDict['features'])
        pcDict = theDict['pcDIct']
        distance_matrix = bfs(N, pcDict)
        attn_bias = np.int32(distance_matrix) 
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, MAXNODE)
        heights = pad_heights(theDict['heights'], MAXNODE)
        if executed:
            gtvalue = pad_2d_unsqueeze(theDict['gtvalue'], MAXNODE)
            return {
            'x': x,
            'attn_bias': attn_bias,
            'heights': heights,
            'gtvalue': gtvalue,
        }
        else:
            return {
            'x': x,
            'attn_bias': attn_bias,
            'heights': heights,
        }

    # def get_column_feature(self, colNames, metaInfo, featureList):
    #     columnFeature = {}
    #     for colName in colNames:
    #         colAttr = metaInfo['colAttr'][colName]
    #         for feature in featureList:
    #             if feature not in columnFeature:
    #                 columnFeature[feature] = []
    #             columnFeature[feature].append(colAttr[feature])
    #     for feature in columnFeature:
    #         columnFeature[feature] = np.array(columnFeature[feature])
    #     return columnFeature
        
    def traverse_node(self, planNode, executed, alias2table, pos = None, parentAlias = None, metaInfo = None, dbName = None):
        try:
            nodeType = planNode['Node Type']
        except:
            print(planNode)
            raise ValueError('Node Type Parse Error')
        table = 'NA'
        alias = None
        if nodeType in SCANTYPE:
            try:
                table = planNode['Relation Name']                
                # tableId = self.table2idx[planNode['Relation Name']]
            except:
                print(f"Relation Name Parse Error: {planNode['Relation Name']}")
                raise ValueError('Relation Name Parse Error')
            try:
                if 'Alias' in planNode:
                    alias = planNode['Alias']
                    if alias not in alias2table.keys():
                        alias2table[planNode['Alias']] = table
            except:
                raise ValueError('Alias Parse Error')
        if alias == None:
            alias = parentAlias  
        filterFeature = {"colName":[], "op":[], "dtype":[],'isInMCV':[],'isInHist':[]}
        joinFeature = []
        parse_conditions(planNode, alias, alias2table, metaInfo['colAttr'], filterFeature, joinFeature)

        # Process the Join Features
        joinColumn = []
        joinMask = []
        for join in joinFeature:
            joinColumn.append(self.assindex['column2idx'][dbName + '.' + join[0]])
            joinColumn.append(self.assindex['column2idx'][dbName + '.' + join[1]])
            joinMask.extend([1,1])
        if len(joinColumn) > MAXJOIN:
            print(f'joinColumn = {len(joinColumn)} > MAXJOIN = {MAXJOIN}, planNode = {planNode}',flush=True)
            joinColumn = joinColumn[:MAXJOIN]
            joinMask = joinMask[:MAXJOIN]
        for _ in range(MAXJOIN - len(joinColumn)):
            joinColumn.append(self.assindex['column2idx']['NA'])
            joinMask.append(0)
        joinEmbed = {'joinColumn':joinColumn, 'joinMask':joinMask}

        # Process the Filter Features
        currentFilterNum = len(filterFeature['colName'])
        for _ in range(MAXFILTER - currentFilterNum):
            filterFeature['colName'].append('NA')
        filterFeature['column'] = []
        for colname in filterFeature['colName']:
            if colname == 'NA':
                filterFeature['column'].append(self.assindex['column2idx'][colname])
            else:   
                filterFeature['column'].append(self.assindex['column2idx'][dbName + '.' + colname])
        del filterFeature['colName']
        try:
            nodeTypeId = TYPE2IDX[nodeType]
        except:
            print(f"NodeType Parse Error: {nodeType}")
            nodeTypeId = len(TYPE2IDX)
        # DBMS Est Info
        # planrows    = minmax_transform(math.log(1 + int(planNode['Plan Rows'])), self.transform_statistics['Plan Rows'])
        # totalcost   = minmax_transform(math.log(1 + int(planNode['Total Cost'])), self.transform_statistics['Total Cost'])
        # planwidth   = minmax_transform(math.log(1 + int(planNode["Plan Width"])), self.transform_statistics['Plan Width'])
        # startupcost = minmax_transform(math.log(1 + int(planNode['Startup Cost'])), self.transform_statistics['Startup Cost'])
        planrows    = math.log10(1 + int(planNode['Plan Rows']))
        totalcost   = math.log10(1 + (int(planNode['Total Cost']) % 1e11)) # avoid the imact of the disable in pg
        planwidth   = math.log10(1 + int(planNode["Plan Width"]))
        startupcost = math.log10(1 + (int(planNode['Startup Cost']) % 1e11))
        dbEst = [planrows, totalcost, planwidth, startupcost]

        if table != 'NA':
            tableEmbed = self.assindex['table2idx'][dbName + '.' + table]
        else:
            tableEmbed = self.assindex['table2idx']['NA']
        gtValue = None
        if executed:
            gtLoops = int(planNode['Actual Loops'])
            gtTime = float(planNode['Actual Total Time']) * gtLoops
            gtRows = float(planNode['Actual Rows']) * gtLoops
            
            gtValue = [gtTime, gtRows] 
        node = TreeNode(tableEmbed, nodeTypeId, joinEmbed, filterFeature, dbEst, pos, gtValue, alias)
        return node
    
    def traverse_plan(self, plan, hint, dbName, executed = False): 
        # pos:{3:'root', 0:'left', 1:'right', 2:'internal-no-brother'}
        alias2table = self.alias2table[dbName]
        metaInfo = self.metaInfo[dbName]
        adj_list = []  
        features = []
        heights = []
        if executed:
            node_gtvalue = []
        root = self.traverse_node(plan, executed, alias2table, pos = 3, metaInfo=metaInfo, dbName=dbName)
        NodeList = deque() 
        NodeList.append((root, plan, '0', 0))
        next_id = 1
        while NodeList:
            parentNode, parentPlan, parentEncod, idx = NodeList.popleft()
            features.append(parentNode.feature)
            heights.append(len(parentEncod))
            if executed:
                node_gtvalue.append(parentNode.gtValue)
            if parentPlan['Node Type'] in JOINTYPE:
                hint['join operator'][parentEncod] = parentPlan['Node Type']
            elif parentPlan['Node Type'] in SCANTYPE:
                hint['scan table'][parentEncod] = [parentPlan['Alias']]
                hint['scan operator'][parentEncod] = parentPlan['Node Type']
            if 'Plans' in parentPlan:
                subPlanNum = len(parentPlan['Plans'])
                if subPlanNum == 1:
                    subplan = parentPlan['Plans'][0]
                    node = self.traverse_node(subplan, executed, alias2table, pos = 2, parentAlias = parentNode.alias, metaInfo=metaInfo, dbName=dbName)
                    subEncod = parentEncod + '0'
                    NodeList.append((node, subplan, subEncod, next_id))
                    adj_list.append((idx, next_id))
                    next_id += 1
                else:
                    for child_idx in range(subPlanNum - 1, -1 , -1):
                        subplan = parentPlan['Plans'][child_idx]
                        node = self.traverse_node(subplan, executed, alias2table, pos = child_idx, parentAlias = parentNode.alias, metaInfo=metaInfo, dbName=dbName)
                        subEncod = parentEncod + str(child_idx)
                        NodeList.append((node, subplan, subEncod, next_id))
                        adj_list.append((idx, next_id))
                        next_id += 1
        pcDIct = defaultdict(list)
        for parent, child in adj_list:
            pcDIct[parent].append(child)
        if len(features) > MAXNODE:
            raise ValueError(f'features = {len(features)} > MAXNODE = {MAXNODE}')
        if executed:
            return {"features": np.float32(features),"heights": np.float32(heights),"pcDIct": pcDIct, "gtvalue": np.float32(node_gtvalue)}
        else:
            return {"features": np.float32(features),"heights": np.float32(heights),"pcDIct": pcDIct}
        
    def get_node_values(self, plan): 
        node_gtvalue = []
        NodeList = deque()
        NodeList.append((plan['Actual Total Time'], plan))
        while NodeList:
            actualNodeTime, parentPlan = NodeList.popleft()
            node_gtvalue.append(math.log10(1 + actualNodeTime) / 7 + 1e-7) 
            if 'Plans' in parentPlan:
                subPlanNum = len(parentPlan['Plans'])
                if subPlanNum == 1:
                    subplan = parentPlan['Plans'][0]
                    NodeList.append((subplan['Actual Total Time'], subplan))
                else:
                    for child_idx in range(subPlanNum - 1, -1 , -1):
                        subplan = parentPlan['Plans'][child_idx]
                        NodeList.append((subplan['Actual Total Time'], subplan))
        node_gtvalue = pad_1d_values(np.array(node_gtvalue), MAXNODE)
        return node_gtvalue

    def _exit(self):
        if self.pghelper:
            self.pghelper.close()
