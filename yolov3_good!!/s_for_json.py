# -*- coding: utf-8 -*-
#from socket import *

#serverSock = socket(AF_INET, SOCK_STREAM)
#serverSock.bind(('', 8080))
#serverSock.listen(1)

#connectionSock, addr = serverSock.accept()

#print(str(addr),'에서 접속이 확인되었습니다.')

#data = connectionSock.recv(1024)
#print('받은 데이터 : ', data.decode('utf-8'))

#connectionSock.send('I am a server.'.encode('utf-8'))
#print('메시지를 보냈습니다.')

from socket import *


port = 9999

serverSock = socket(AF_INET, SOCK_STREAM) #소켓 생성
serverSock.bind(('', port)) #Bind
serverSock.listen(1) #클라이언트와 연결 대기

print('%d번 포트로 접속 대기중...'%port)

connectionSock, addr = serverSock.accept() #클라이언트와 연결 성공

print(str(addr), '에서 접속되었습니다.')

while True:
    #sendData = input('>>>')
    #connectionSock.send(sendData.encode('utf-8'))
    recvData = connectionSock.recv(1024) #클라이언트로부터 데이터 받음 Receive
    decRecvData = recvData.decode('utf-8')
    print('Client :', decRecvData) #클라이언트로부터 받은 데이터 해독
    if decRecvData == 'exit':
        break
