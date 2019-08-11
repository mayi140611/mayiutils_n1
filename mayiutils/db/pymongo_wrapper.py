#!/usr/bin/python
# encoding: utf-8
# version: pymongo==3.7.1

#待处理：
#db.getCollection('symptomsdetail').find({}).sort({'titlepy':1})
from pymongo import MongoClient
import pandas as pd
from collections import namedtuple
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper

class PyMongoWrapper(object):
    def __init__(self, ip='localhost', port=27017):
        self._client = MongoClient(ip, port)#MongoClient('localhost', 27017) or MongoClient('mongodb://localhost:27017/')
    
    def getDb(self, dbname):
        return self._client[dbname]

    def getCollection(self, dbname, collection):
        return self.getDb(dbname)[collection]
    
    def setUniqueIndex(self, dbname, collection, field):
        '''
        为单一字段设置唯一索引
        '''
        try:
            self.getDb(dbname)[collection].ensure_index(field, unique=True)
            return True
        except Exception as e:
            print(e)
            return False
    def findAll(self, collection, conditions=None, fieldlist='all'):
        '''
        查找所有数据，返回指定的fieldlist
        :conditions 查询条件。
            {'c1':'全身'}
            {'c2':{'$exists':False}}：把不存在某个属性的行都查出来的条件
            {'$and/or': [ { <expression1> }, { <expression2> } , ... , { <expressionN> } ] }
        :fieldlist 'all'表示返回所有数据，或者是一个字段list
        '''
        d = dict()
        if fieldlist != 'all':
            if '_id' not in fieldlist:
                d['_id'] = 0
            for i in fieldlist:
                d[i] = 1
            return collection.find(conditions, d)
        return collection.find(conditions)

    def findOne(self, collection, conditions=None, fieldlist='all'):
        '''
        查找所有数据，返回指定的fieldlist
        :conditions 查询条件。{'c1':'全身'}
            注：把不存在某个属性的行都查出来的条件{'c2':{'$exists':False}}
        :fieldlist 'all'表示返回所有数据，或者是一个字段list
        '''
        d = dict()
        if fieldlist != 'all':
            if '_id' not in fieldlist:
                d['_id'] = 0
            for i in fieldlist:
                d[i] = 1
            return collection.find_one(conditions, d)
        return collection.find_one(conditions)

    def updateDoc(self, collection, conditions,fielddict):
        '''
        更新表中的符合条件的第一条记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :conditions:如{'c1':'v1','c2':'v2'}
        :fielddict: 如{'f1':'v1','f2':'v2'}
        '''
        return collection.update_one(conditions,{ "$set": fielddict})
    def updateDocs(self, collection, conditions,fielddict):
        '''
        更新表中符合条件的所有记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :conditions:如{'c1':'v1','c2':'v2'}
        :fielddict: 如{'f1':'v1','f2':'v2'}
        '''
        return collection.update_many(conditions,{ "$set": fielddict})
    def deleteDoc(self, collection, conditions):
        '''
        删除符合条件的一个文档
        :param collection:
        :param conditions: 过滤条件
            {'_id': i['_id']}
        :return:
        '''
        return collection.delete_one(conditions)

    def removeDocFields(self, collection, conditions,fieldslist):
        '''
        更新表中符合条件的一条记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :fieldslist: 如['f1','f2'...]
        '''
        return collection.update_one(conditions,{ "$unset": {ii:"" for ii in fieldslist}})

    def removeDocsFields(self, collection, conditions,fieldslist):
        '''
        更新表中符合条件的所有记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :fieldslist: 如['f1','f2'...]
        '''
        return collection.update_many(conditions,{ "$unset": {ii:"" for ii in fieldslist}})


if __name__ == '__main__':
    mysql = PyMysqlWrapper(host='h1')
    mode = 2
    if mode == 2:
        """
        存储专科医院表
        """
        pmw = PyMongoWrapper('h1')
        table = pmw.getCollection('intelligent-guidance', 'specialhosdeptmapping')
        #专科医院和其对应的科室
        dict1 = {'东方肝胆外科医院(杨浦院区)': '肝病科、肝胆外科',
         '儿科医院': '儿科',
         '儿童医学中心': '儿科',
         '儿童医院(北京西路院区)': '儿科',
         '龙华医院': '中医科',
         '儿童医院(泸定路院区)': '儿科',
         '上海市中医医院': '中医科',
         '上海市中医医院(石门路门诊部)': '中医科',
         '同济口腔医院': '口腔科',
         '皮肤病医院(武夷路院区)': '皮肤性病科',
         '眼病防治中心': '眼科',
         '口腔病防治院(分院)': '眼科',
         '胸科医院': '心胸外科',
         '皮肤病医院(保德路院区)': '皮肤性病科',
         '精神卫生中心': '精神心理科、精神科、心理咨询科',
         '口腔病防治院(总院)': '口腔科',
         '曙光医院(西院)': '中医科',
         '曙光医院(东院)': '中医科',
         '肿瘤医院': '肿瘤科',
         '岳阳中西医结合医院(青海路名医特诊部)': '中医科',
         '上海市第一妇婴保健院(东院)': '妇产科',
         '上海市第一妇婴保健院(西院)': '妇产科',
         '肺科医院': '呼吸内科',
         '肺科医院(徐汇区延庆路门诊部)': '呼吸内科',
         '妇产科医院(杨浦院区)': '妇产科',
         '妇产科医院(黄浦院区)': '妇产科',
         '岳阳中西医结合医院': '中西医结合科',
         '眼耳鼻喉科医院(含宝庆浦江)': '眼科、耳鼻喉科'}
        sqltemplate = """
            SELECT h.hospitalAdd as addr,
                h.hosOrgCode as hoscode,
                h.latLng as latlng,
                h.hospitalGrade as grade
            from kh_hospital h
            where h.hosName='{}'
            """
        list1 = ['addr', 'hoscode', 'latlng', 'grade']
        for k, v in dict1.items():
            mysql.execute(sqltemplate.format(k))
            data = mysql._cursor.fetchone()
            d = {'hosname': k, 'dept': v.split('、')}
            d.update(dict(zip(list1, data)))
            table.insert_one(d)

    if mode == 1:
        sqltemplate = """
            SELECT
                h.hosOrgCode AS hoscode,
                h.hospitalAdd AS addr,
                h.latLng as latLng,
                d.hosDeptCode AS deptcode,
                d.topHosDeptCode AS topdeptcode,
                d.deptType as type,
                d.hosName as yyname,
                d.deptName as ksname
            FROM
                kh_hospital h,
                kh_dept d
            WHERE h.hosOrgCode=d.hosOrgCode and  d.hosName = '{}'
                AND d.deptName = '{}'
            """

        pmw = PyMongoWrapper('h1')
        table = pmw.getCollection('intelligent-guidance', 'deptmapping')
        # for i in pmw.findAll(table, fieldlist=['简介']):
        #     print(' '.join(i['简介']))
        df = pd.read_excel('../../tmp/deptmapping.xlsx', '申康可预约科室')
        # print(df.head())
        # print(df.loc[:, '医院类型'].unique())#['专科医院' '专科医院 ' '中医医院' '综合医院' '综合医院 ']
        # print(df.loc[:, '专病'].unique())
        """
        ['肝病科、肝胆外科' '儿科' '中医科' nan '口腔科' '皮肤性病科' '眼科' '心胸外科' '精神心理科、精神科、心理咨询科'
     '肿瘤科' '妇产科' '呼吸内科' '呼吸内科、心胸外科' '皮肤科' '肝胆外科、肝病科' '中西医结合科' '眼科、耳鼻喉科']
        """
        dict1 = dict()
        for line in df[df.loc[:, '医院类型'].isin(['专科医院', '专科医院 ', '中医医院'])].itertuples():
            if line[1] not in dict1:
                dict1[line[1]] = line[7]
        # print(dict1)
        """
        {'东方肝胆外科医院(杨浦院区)': '肝病科、肝胆外科', 
        '儿科医院': '儿科', 
        '儿童医学中心': '儿科', 
        '儿童医院(北京西路院区)': '儿科', 
        '龙华医院': '中医科', 
        '儿童医院(泸定路院区)': '儿科', 
        '上海市中医医院': '中医科', 
        '上海市中医医院(石门路门诊部)': '中医科', 
        '同济口腔医院': '口腔科', 
        '皮肤病医院(武夷路院区)': '皮肤性病科', 
        '眼病防治中心': '眼科', 
        '口腔病防治院(分院)': '眼科', 
        '胸科医院': '心胸外科', 
        '皮肤病医院(保德路院区)': '皮肤性病科', 
        '精神卫生中心': '精神心理科、精神科、心理咨询科', 
        '口腔病防治院(总院)': '口腔科', 
        '曙光医院(西院)': '中医科', 
        '曙光医院(东院)': '中医科',
         '肿瘤医院': '肿瘤科', 
         '岳阳中西医结合医院(青海路名医特诊部)': '中医科', 
         '上海市第一妇婴保健院(东院)': '妇产科', 
         '上海市第一妇婴保健院(西院)': '妇产科', 
         '肺科医院': '呼吸内科', 
         '肺科医院(徐汇区延庆路门诊部)': '呼吸内科', 
         '妇产科医院(杨浦院区)': '妇产科', 
         '妇产科医院(黄浦院区)': '妇产科', 
         '岳阳中西医结合医院': '中西医结合科', 
         '眼耳鼻喉科医院(含宝庆浦江)': '眼科、耳鼻喉科'}
        """
        # print(df.loc[:, '备注'].unique())
        """
        [nan '0-14岁' '0-1岁' '女' '15岁及以上' '0-15岁' '0-16岁' '产褥期' '孕期' '注意' '不可约'
     '骨质疏松' '鼻炎' '哮喘' '呼吸道感染' '尿失禁' '糖尿病' '肾炎' '高血压' '糖尿病神经病变' '糖尿病肾病' '孕妇'
     '高血压高血脂' '糖尿病护理']
        """
        Item = namedtuple('Item', ['hosname', 'dept', 'aidept', 'age', 'gender', 'other'])
        list1 = ['hoscode', 'addr', 'latLng', 'deptcode', 'topdeptcode', 'type']
        for line in df[df.loc[:, '医院类型'].isin(['综合医院', '综合医院 '])].itertuples():
            if line[4] and str(line[4]).strip():
                if str(line[6]).strip() not in ['注意', '不可约', '骨质疏松', '鼻炎', '哮喘', '呼吸道感染', '尿失禁', '糖尿病', '肾炎', '高血压',
                                                '糖尿病神经病变', '糖尿病肾病', '高血压高血脂', '糖尿病护理']:
                    # print(line[1], line[2], line[4], line[6])
                    if str(line[6]).strip() in ['0-14岁', '0-15岁', '0-16岁']:
                        r = Item(line[1], line[2], line[4], [2, 3, 4], [0, 1], [0, 1, 2])
                    elif str(line[6]).strip() == '女':
                        r = Item(line[1], line[2], line[4], [1, 2, 3, 4, 5, 6, 7], [1], [0, 1, 2])
                    elif str(line[6]).strip() == '产褥期':
                        r = Item(line[1], line[2], line[4], [1, 2, 3, 4, 5, 6, 7], [0, 1], [2])
                    elif str(line[6]).strip() in ['孕期', '孕妇']:
                        r = Item(line[1], line[2], line[4], [2, 3, 4], [0, 1], [1])
                    else:
                        r = Item(line[1], line[2], line[4], [1, 2, 3, 4, 5, 6, 7], [0, 1], [0, 1, 2])
                    print(r)
                    # print(dict(zip(['hosname', 'dept', 'aidept', 'age', 'gender', 'other'], list(r))))
                    # table.insert_one(dict(zip(['hosname', 'dept', 'aidept', 'age', 'gender', 'other'], list(r))))
                    mysql.execute(sqltemplate.format(line[1], line[2]))
                    data = mysql._cursor.fetchone()
                    if data:
                        dict1 = dict(zip(list1, data))
                        print(dict1)
                        dict2 = dict(zip(['hosname', 'dept', 'aidept', 'age', 'gender', 'other'], list(r)))
                        print(dict2)
                        dict1.update(dict2)
                        table.insert_one(dict1)
                    print(data)
                    # break
