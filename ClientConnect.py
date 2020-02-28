from socket import socket, AF_INET, SOCK_STREAM
import struct

class Client():
    def __init__(self):
        self.connectingSocket = socket(AF_INET, SOCK_STREAM)
        self.connectingSocket.setblocking(True)

    def connect_to_server(self, ip, port, name = 'ConnectionTest'):
        self.connectingSocket.connect((ip, port))
        connection_template = 'Connected to %s:%s with name %s'
        print(connection_template % (ip, port, name))
        return self.send_message(['NME', len(name), name])

    def fetch_data(self, size, byte_header):
        recv = bytes()
        while len(recv) < size:
            recv += self.connectingSocket.recv(size - len(recv))
        return struct.unpack(byte_header, recv)
    
    def get_message_type(self):
        return self.connectingSocket.recv(3).decode('ascii')

    def get_message(self, expected):
        msg_type = self.get_message_type()
        print('Message type '+msg_type)
        switcher = {
            'SET': self.set_message,
            'HUM': self.hum_message,
            'HME': self.hme_message,
            'MAP': self.map_message,
            'UPD': self.upd_message,
            'END': self.end_message,
            'BYE': self.bye_message
        }
        if msg_type == expected:
            action = switcher.get(msg_type, lambda: self.error)
            return action()
        else:
            print("Encountered error processing message, expected: "+ expected +" but got "+msg_type)

    def send_message(self, payload):
        if payload[0]=="NME":
            self.connectingSocket.send(payload[0].encode("ascii"))
            self.connectingSocket.send(struct.pack("1B",payload[1]))
            self.connectingSocket.send(payload[2].encode("ascii"))
            return self.get_message("SET")
        elif payload[0]=="MOV":
            self.connectingSocket.send(payload[0].encode("ascii"))
            n=payload[1]
            self.connectingSocket.send(struct.pack("1B",n))
            self.connectingSocket.send(struct.pack("{}B".format(5*n), *(payload[2])))
            return self.get_message("UPD")

    def set_message(self):
        n,m=self.fetch_data(2,"2B")
        return [(n,m)]+self.get_message("HUM")

    def hum_message(self):
        res=[]
        n= self.fetch_data(1,"1B")[0]
        homes= self.fetch_data(2*n,"{}B".format(2*n))
        count=0
        prev=0
        for h in homes:
            if count%2==0:
                prev=h
            else:
                res.append((prev,h))
            count+=1
        return [n]+res+self.get_message("HME")

    def hme_message(self):
        start_pos=tuple(self.fetch_data(2,"2B"))
        return [start_pos]+ self.get_message('MAP')

    def map_message(self):
        n=self.fetch_data(1,"1B")[0]
        commands=self.fetch_data(5*n,"{}B".format(5*n))
        res=[]
        x=0
        y=0
        h=0
        v=0
        count=0
        for c in commands:
            if count%5==0:
                x=c
            elif count%5==1:
                y=c
            elif count%5==2:
                h=c
            elif count%5==3:
                v=c
            else:
                res.append((x,y,h,v,c))
            count+=1
        return res

    def upd_message(self):
        n=self.fetch_data(1,"1B")[0]
        commands=self.fetch_data(5*n,"{}B".format(5*n))
        res=[]
        x=0
        y=0
        h=0
        v=0
        count=0
        for c in commands:
            if count%5==0:
                x=c
            elif count%5==1:
                y=c
            elif count%5==2:
                h=c
            elif count%5==3:
                v=c
            else:
                res.append((x,y,h,v,c))
            count+=1
        return res

    def end_message(self):
        print("Game over, resetting data . . .")
        return 1

    def bye_message(self):
        print("Connection closing, bye . . .")
        return 2

    def error(self):
        print('Undocumented error')
        return 0

