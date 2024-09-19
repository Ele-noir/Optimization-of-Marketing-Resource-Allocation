from enum import Flag
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time
import numpy as np
import requests
import simplejson as json
import csv
import io


# Save Comment Data
def commentSave(list_comment):
    '''
    list_comment: 2d list, includes multiple user comments
    '''
    file = io.open('/Users/eleanor.z/Desktop/JDComment_Spider/Herborist_JDComment_data.csv','w',encoding="utf-8",newline = '')
    writer = csv.writer(file)
    writer.writerow(['用户ID','评论内容','购买时间','点赞数','回复数','得分','评价时间','护肤水型号'])
    for i in range(len(list_comment)):
        writer.writerow(list_comment[i])
    file.close()
    print('存入成功')

# Crawl for Comments in the Webpage
def getCommentData(format_url,proc,i,maxPage):
    '''
    format_url: Formatted string shelf, append parameters to it in the loop.
    proc: productID, which identifies the unique product number
    i: The way the goods are sorted, such as all the goods, printing pictures, follow-up reviews, praise, etc
    maxPage: The maximum number of pages of reviews for an item
    '''
    sig_comment = []
    global list_comment
    cur_page = 0
    while cur_page < maxPage:
        cur_page += 1
        url = format_url.format(proc,i,cur_page) # Append parameters to the string
        try:
            response = requests.get(url=url, headers=headers, verify=False)
            time.sleep(np.random.rand()*2)
            jsonData = response.text
            startLoc = jsonData.find('{')
            #print(jsonData[::-1])// reverse the string
            jsonData = jsonData[startLoc:-2]
            jsonData = json.loads(jsonData)
            pageLen = len(jsonData['comments'])
            print("当前第%s页"%cur_page)
            for j in range(0,pageLen):
                userId = jsonData['comments'][j]['id']#User ID
                content = jsonData['comments'][j]['content']#Review Comments
                boughtTime = jsonData['comments'][j]['referenceTime']#Purchase Time
                voteCount = jsonData['comments'][j]['usefulVoteCount']#Likes
                replyCount = jsonData['comments'][j]['replyCount']#Number of replies
                starStep = jsonData['comments'][j]['score']#Score
                creationTime = jsonData['comments'][j]['creationTime']#Comment Time
                referenceName = jsonData['comments'][j]['referenceName']#Item Type
                sig_comment.append(userId)
                sig_comment.append(content)
                sig_comment.append(boughtTime)
                sig_comment.append(voteCount)
                sig_comment.append(replyCount)
                sig_comment.append(starStep)
                sig_comment.append(creationTime)
                sig_comment.append(referenceName)
                list_comment.append(sig_comment)
                print(sig_comment)
                sig_comment = []
        except:
            time.sleep(5)
            cur_page -= 1
            print('网络故障或者是网页出现了问题，五秒后重新连接')



if __name__ == "__main__":
    global list_comment
    ua=UserAgent()
    format_url='https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&body=%7B%22productId%22%3A{0}%2C%22score%22%3A{1}%2C%22sortType%22%3A5%2C%22page%22%3A{2}%2C%22pageSize%22%3A10%2C%22isShadowSku%22%3A0%2C%22rid%22%3A0%2C%22fold%22%3A1%2C%22bbtf%22%3A%22%22%2C%22shield%22%3A%22%22%7D'
    # Set the access request header
    headers = {
    'Accept': '*/*',
    'Host':"club.jd.com",
    "User-Agent":ua.random,
    'sec-ch-ua':"\"Chromium\";v=\"92\", \" Not A;Brand\";v=\"99\", \"Google Chrome\";v=\"92\"",
    'sec-ch-ua-mobile': '?0',
    'Sec-Fetch-Dest': 'script',
    'Sec-Fetch-Mode':'no-cors',
    'Sec-Fetch-Site':'same-site',
    }
    
    #Product id parameter
    productid = ['100018304301','100088855329','100014592543','100032432899','100012572326','100002221494','100067971978']
    list_comment = [[]]
    sig_comment = []
    for proc in productid:#Traverse different productid
        i = -1
        while i < 7:#Traverse all sorting
            i += 1
            if(i == 6):
                continue
             #First access page 0 for the maximum number of pages, and then loop through
            url = format_url.format(proc,i,0)
            print(url)
            try:
                response = requests.get(url=url, headers=headers, verify=False)
                jsonData = response.text
                print(jsonData)
                startLoc = jsonData.find('{')
                jsonData = jsonData[startLoc:-2]
                jsonData = json.loads(jsonData)
                print("最大页数%s"%jsonData['maxPage'])
                getCommentData(format_url,proc,i,jsonData['maxPage'])#Traverse through all pages
            except Exception as e:
                i -= 1
                print("the error is ",e)
                print("waiting---")
                time.sleep(5)
                #commentSave(list_comment)
    print("爬取结束，开始存储-------")
    
    commentSave(list_comment)