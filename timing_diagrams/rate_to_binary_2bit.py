import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",             "wave": "P.......", "period": 2  },
 { "name": "R",               "wave": "l.hl..hl........" },
 { "name": "A",               "wave": "l.........hl..hl","node":'a..........b..c.' },
 { "name": "V2-left (0x01)",  "wave": "l.....hl........"},
 { "name": "V2-right (0x10)", "wave": "l..............."},
 { "name": "P",               "wave": "l.........hl...." },
 { "name": "T0-left",         "wave": "l..............." },
 { "name": "T0-right",        "wave": "l.........hl...." },
],
 "edge":['a-~>b D5', 'a-~>c D7']

}""")
svg.saveas("rate_to_binary_2bit.svg")
