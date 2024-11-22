import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",         "wave": "P........", "period": 2  },
 { "name": "R",           "wave": "l.hlhlhl..hlhl...." },
 { "name": "A",           "wave": "l.....hl....hl....", "data":['1']},
 { "name": "0x001",       "wave": "2...2.2.2...2.2...", "data":['0','1','0','1','0','1']},
 { "name": "B",           "wave": "l.............hl.."},
 { "name": "0x010",       "wave": "2.......2.....2...", "data":['0','1','0']},
 { "name": "C",           "wave": "l................." },
 { "name": "0x100",       "wave": "2...............2.", "data":['0','1']},
],
}""")
svg.saveas("rate_to_binary_subcircuit.svg")
