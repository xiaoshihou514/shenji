import chess.pgn
import json
import os

def main():
    pgn = open("lichess_db_standard_rated_2025-05.pgn")
    max = 100000

    i: int = 0
    j: int = 0
    while (game := chess.pgn.read_game(pgn)):
        t = {}
        hs = game.headers
        if int(hs["WhiteElo"]) >= 2000 and int(hs["BlackElo"]) >= 2000:
            j += 1
            t["result"] = hs["Result"]
            t["moves"] = []
            for move in game.mainline_moves():
                t["moves"].append(move.uci())
            path = f'processed/{hs["UTCDate"]}-{hs["UTCTime"]}-{hs["White"]}vs{hs["Black"]}'
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump(t, f)
            print(f"{j}/{i}")
        if j > max:
            return
        i += 1

if __name__ == "__main__":
    main()

