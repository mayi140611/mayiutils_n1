#!/usr/bin/python
# encoding: utf-8
# version: pymongo==3.7.1

#待处理：
#db.getCollection('symptomsdetail').find({}).sort({'titlepy':1})
from pymongo import MongoClient


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
    def findAll(self, collection, conditions=None, fieldlist='all', sort=False, limit=False):
        '''
        查找所有数据，返回指定的fieldlist
        :conditions 查询条件。
            {'c1':'全身'}
            {'c2':{'$exists':False}}：把不存在某个属性的行都查出来的条件
            {'$and/or': [ { <expression1> }, { <expression2> } , ... , { <expressionN> } ] }
            {'trade_date': {'$lte': datetime(2001, 1, 5)}}
            {'$and': [{'trade_date':{'$lte': datetime(2001, 1, 5)}}, {'trade_date':{'$gt': datetime(2001, 1, 2)}}]}

        :fieldlist 'all'表示返回所有数据，或者是一个字段list
        :sort: list  1代表升序， -1代表降序
            [(field1,-1), (field2,1)]
        :limit: positive integer
        '''
        d = dict()
        if fieldlist != 'all':
            if '_id' not in fieldlist:
                d['_id'] = 0
            for i in fieldlist:
                d[i] = 1
            r = collection.find(conditions, d)
        else:
            r = collection.find(conditions)
        if sort:
            r = r.sort(sort)
        if limit:
            r = r.limit(limit)
        return r

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


