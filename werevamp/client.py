from socket import socket, AF_INET, SOCK_STREAM
import struct
import argparse, sys
from game import Game

class Client():
    def __init__(self):
        self.connectingSocket = socket(AF_INET, SOCK_STREAM)
        self.connectingSocket.setblocking(True)
        self.game_state_init = dict()

    def intialize_game_state(self, ip, port, name = 'ConnectionTest'):
        self.connectingSocket.connect((ip, port))
        print(f'Connected to {ip}:{port} with name {name}')
        game_state_init = self.send_message(['NME', len(name), name])
        return game_state_init

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
            print(f"Encountered error processing message, expected: {expected} but got {msg_type}")

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
        self.game_state_init['boardsize'] = (n,m)
        return self.get_message("HUM")

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
        self.game_state_init['house_coords'] = res
        return self.get_message("HME")

    def hme_message(self):
        start_pos=tuple(self.fetch_data(2,"2B"))
        self.game_state_init['starting_tile'] = start_pos
        return self.get_message('MAP')

    def map_message(self):
        n=self.fetch_data(1,"1B")[0]
        commands=self.fetch_data(5*n,"{}B".format(5*n))
        res=dict()
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
                res[(x,y)] = (h,v,c)
            count+=1
        self.game_state_init['map_initialization'] = res
        self.game_state_init['race'] = self._get_starting_race()
        self._parse_game_state()
        return self.game_state_init

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
        del self.game
        del self.game_state_init
        return 1

    def bye_message(self):
        print("Connection closing, bye . . .")
        del self.game
        del self.game_state_init
        return 2

    def error(self):
        print('Undocumented error')
        return 3
    
    def _get_starting_race(self):
        mapping = self.game_state_init['map_initialization']
        if mapping is None:
            raise ValueError('Map is not initialized')
        relevant_tile = mapping[self.game_state_init['starting_tile']]
        race = None
        for i, elt in enumerate(relevant_tile):
            if elt > 0 and race is None:
                race = i
            elif elt > 0 and race:
                raise ValueError('More than one race in one tile, should not happen.')
        return race

    def _parse_game_state(self):
        def find_occupying_race(tile_list, mapping):
            """
            Returns the given occupying race of a tile based on Game terminology
            Example:
                Given a list (0, 3, 0), returns 2 as the tile is occupied by 3 vampires.
            """
            assert len(tile_list) == 3
            for i, val in enumerate(tile_list):
                if val > 0 : 
                    return (mapping[i], val)

        n,m = self.game_state_init['boardsize']
        self.game = Game(n,m)
        mapping = [Game.Human, Game.Vampire, Game.Werewolf]
        self.playing_race = mapping[self.game_state_init['race']]
        for key, value in self.game_state_init['map_initialization'].items():
            self.game[key[1], key[0]] = find_occupying_race(value, mapping)
        return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', help='Provide port for server connection')
    parser.add_argument('--ip', help='Provide ip address for server connection')
    parser.add_argument('--name', help='Player name')
    args = parser.parse_args()
    if args.port:
        port = int(args.port)
    else:
        port = 5555
    if args.ip:
        ip = args.ip
    else:
        ip = "127.0.0.1"
    if args.name:
        name = args.name
    else:
        name = 'player'
    
    client = Client()
    res = client.intialize_game_state(ip, port, name)
    if res in (1,2,3):
        print("error")
    else:
        print(res)
        print(client.game)
        print(client.playing_race)
