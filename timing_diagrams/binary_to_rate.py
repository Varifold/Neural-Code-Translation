import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",             "wave": "P.....", "period": 2  },
 { "name": "MSb-B",           "wave": "hl.........." },
 { "name": "MSb-V1",          "wave": "l.hl..hl....", "node":'a.b...c'},
 { "name": "MSb-V2",          "wave": "l.....hl...." },
 { "name": "LSb-B",           "wave": "hl.........." },
 { "name": "LSb-V1-left",     "wave": "l...hl......", "node":'d...e.......f'},
 { "name": "LSb-V1-right",    "wave": "l.........hl", "node":'....g.....h'},
 { "name": "P",               "wave": "l.hlhlhl...." },
],
"edge":['a-~>b D1', 'b-~>c D2', 'd-~>e D2', 'e-~>f D4', 'g-~>h D3']
}""")
svg.saveas("binary_to_rate.svg")
