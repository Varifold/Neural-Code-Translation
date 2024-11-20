import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",    "wave": "P........","period": 2  },
 { "name": "B",     "wave":  "hl................" },
 { "name": "V1",     "wave": "l.hl..hl..hl..hl..", "node":'..a...b...c...d...e'},
 { "name": "V4",  "wave":    "l...............hl", "node":'..............f.g' }
 ],
 "edge":['a-~>b D2', 'b-~>c D2', 'c-~>d D2', 'f-~>g D1']
}""")
svg.saveas("pulse_cell.svg")
