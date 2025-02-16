import chess
import sys
import logging
import queue
import os
from concurrent.futures import ThreadPoolExecutor
from engine import Engine
from Evaluate import CNNValuator
from search_v5 import Searchv5

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(filename="uci.log", level=logging.DEBUG)

class UCI():

    def __init__(self):
        self.board = chess.Board()
        self.engine = Engine(CNNValuator(os.path.join(PROJECT_ROOT, "models/cnn.pth")), Searchv5())
        self.running = True
        self.command_queue = queue.Queue()
        pass

    def read_input(self):
        """Continuosly reads input from stdin and adds it to the queue"""

        while self.running:
            try:
                command = sys.stdin.readline().strip()
                if command:
                    self.command_queue.put(command)
            except Exception as e:
                logging.error(f"Error reading input: {e}")

    def consumer(self):
        """Processes commands from the queue"""
        
        while self.running:
            try:
                command = self.command_queue.get(timeout=1)
                if command:
                    self.ProcessCommand(command)
            except queue.Empty:
                pass

    def ProcessCommand(self, command):
        """Handles UCI commands recieved from GUI"""
        
        logging.info(f"Command recieved: {command}")
        words = command.strip().split(" ")

        if words[0] == "uci":
            self.uci()
        elif words[0] == "setoption":
            self.setoption()
        elif words[0] == "isready":
            self.isready()
        elif words[0] == "ucinewgame":
            self.ucinewgame()
        elif words[0] == "position":
            self.position(words)
        elif words[0] == "go":
            self.go(words)
        elif words[0] == "stop":
            self.stop()
        elif words[0] == "quit":
            self.quit()

    def send_command(self, command):
        """Send command to stdout"""

        logging.debug(f"Command sent: {command}")
        sys.stdout.write(f"{command}\n")
        sys.stdout.flush()

    def uci(self):
        self.send_command("id name nebuliser")
        self.send_command("id author Ben Harvey")
        self.send_command("uciok")

    def isready(self):
        self.send_command("readyok")

    def setoption(self):
        logging.warning("setoption ignored")

    def ucinewgame(self):
        self.board.reset()
    
    def position(self, words):
        """ position [fen <fenstring> | startpos ]  moves <move1> .... <movei>
    set up the position described in fenstring on the internal board and
    play the moves on the internal chess board. if the game was played  from the start position 
    the string "startpos" will be sent.
    """
        
        if words[1] == "startpos" and words[2] == "moves":
            self.board.reset()
            [self.board.push_uci(move) for move in words[3:]]
        elif words[1] == "fen":
            fen = " ".join(words[2:8])
            self.board = chess.Board(fen)
            if len(words) >= 9 and words[8] == "moves":
                [self.board.push_uci(move) for move in words[9:]]


    def go(self, words):
        """Handle 'go' command and extracts search parameters"""

        search_params = {
            "wtime": None,
            "btime": None,
            "winc": None,
            "binc": None,
            }
        
        i = 1
        while i < len(words):
            if words[i] == "wtime":
                search_params["wtime"] = float(words[i+1])
            elif words[i] == "btime":
                search_params["btime"] = float(words[i+1])
            elif words[i] == "winc":
                search_params["winc"] = float(words[i+1])
            elif words[i] == "binc":
                search_params["binc"] = float(words[i+1])
            
            i += 2
        
        colour = self.board.turn
        time_key = "wtime" if colour == chess.WHITE else "btime"
        inc_key = "winc" if colour == chess.WHITE else "binc"

        time = search_params.get(time_key, 500000) / 1000  
        increment = search_params.get(inc_key, 0) / 1000 

        time_for_move = self.engine.time_for_move(time, increment)

        best_move = self.engine.move(self.board, colour, time_for_move)
        self.send_command(f"bestmove {best_move}")

    def quit(self):
        self.running = False
        logging.debug("Engine exit")
        sys.exit(0)

    def stop(self):
        pass 

    def main(self):
        """Starts input reading and command processing in separate threads."""

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.read_input)
            executor.submit(self.consumer)

if __name__ == "__main__":
    uci_engine = UCI()
    uci_engine.main()