from __future__ import absolute_import, division, print_function

import numpy as np
import MA.CSV3D as MA3D
import MA.MeshIO as MAIO
import MA.Tools as MATL

NodesO = MAIO.CNodes(np.array([[0,0,0.0],[1,0,1.1],[2,0,2.3],[0,1,1.2],[1,1,2.0],[2,1,3.1],[0,2,2.1],[1,2,3.3],[2,2,4.7]]))
ElementsO = MAIO.CElements(np.array([[0,1,4],[0,4,3],[1,2,4],[2,5,4],[3,4,6],[4,7,6],[4,5,8],[4,8,7]]))
MyM = MAIO.MyMesh(NodesO,ElementsO)

MyM.ComputeNormals()
MyM.ComputeCurvatures()
MyM.ComputeCurvaturePrincipalDirections()

MyM.Nodes.AppendField(MyM.NMaxCd[:,0],'nx')
MyM.Nodes.AppendField(MyM.NMaxCd[:,1],'ny')
MyM.Nodes.AppendField(MyM.NMaxCd[:,2],'nz')

# Ply = MAIO.PLYIO()
# Ply.ImportMesh(MyM.Nodes,MyM.Elems)
# Ply.SaveFile("test.ply",asASCII=False)

# Vtu = MAIO.VTUIO()
# Vtu.ImportMesh(MyM.Nodes,MyM.Elems)
# Lab=["Curvature",["nx","ny","nz"]]
# Vtu.VecLabs.append(Lab)
# Vtu.SaveFile("test.vtu")
print(MyM.InterpolateFieldForPoint([1.2,1.7],'nx'))

