from test_kpartition_refinement import _assignments

test = _assignments(['a', 'b', 'c'], 3)
for t in test:
  print(t)