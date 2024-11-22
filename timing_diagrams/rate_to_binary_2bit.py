import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",             "wave": "P.......", "period": 2  },
 { "name": "R",               "wave": "l.hlhlhl........" },
 { "name": "A",               "wave": "hl..............","node":'a.b.......c...d.' },
 { "name": "V2-left (0x01)",  "wave": "l.....hl........"},
 { "name": "V2-right (0x10)", "wave": "l..............."},
 { "name": "P",               "wave": "hl...............", "node":'e...........f' },
 { "name": "T0-left",         "wave": "l...........hl.." },
 { "name": "T0-right",        "wave": "l...........hl.." },
],
 "edge":['a-~>b D1', 'b-~>c D5', 'c-~>d D7', 'e-~>f D5']

}""")
svg.saveas("rate_to_binary_2bit.svg")
