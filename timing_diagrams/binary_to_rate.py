import wavedrom
svg = wavedrom.render("""
{ "signal": [
 { "name": "CLK",             "wave": "P.....", "period": 2  },
 { "name": "MSb-B",           "wave": "hl.........." },
 { "name": "MSb-V1",          "wave": "l.hl..hl....", "node":'a.b...c'},
 { "name": "MSb-V2",          "wave": "l.......hl..", "node":'......d.e'},
 { "name": "LSb-B",           "wave": "hl.........." },
 { "name": "LSb-V1-left",     "wave": "l...hl......", "node":'f...g.......h'},
 { "name": "LSb-V1-right",    "wave": "l.........hl", "node":'....i.....j'},
 { "name": "P",               "wave": "l.hlhlhl...." },
],
"edge":['a-~>b D1', 'b-~>c D2', 'd-~>e D1', 'f-~>g D2', 'g-~>h D4', 'i-~>j D3']
}""")
svg.saveas("binary_to_rate.svg")
