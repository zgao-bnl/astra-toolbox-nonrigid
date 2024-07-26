/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#include "astra/CudaBackProjectionAlgorithm3D_DF.h"

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/CompositeGeometryManager.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaBackProjectionAlgorithm3D_DF::type = "BP3D_CUDA_DF";

//----------------------------------------------------------------------------------------
// Constructor
CCudaBackProjectionAlgorithm3D_DF::CCudaBackProjectionAlgorithm3D_DF() 
{
	m_bIsInitialized = false;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_bSIRTWeighting = false;
	m_pDeformX = 0;
	m_pDeformY = 0;
	m_pDeformZ = 0;
}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaBackProjectionAlgorithm3D_DF::CCudaBackProjectionAlgorithm3D_DF(CProjector3D* _pProjector, 
								   CFloat32ProjectionData3D* _pProjectionData, 
								   CFloat32VolumeData3D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaBackProjectionAlgorithm3D_DF::~CCudaBackProjectionAlgorithm3D_DF() 
{
	CReconstructionAlgorithm3D::_clear();
}


//---------------------------------------------------------------------------------------
// Check
bool CCudaBackProjectionAlgorithm3D_DF::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm3D::_check(), "BP3D_CUDA_DF", "Error in ReconstructionAlgorithm3D initialization");


	return true;
}

//---------------------------------------------------------------------------------------
void CCudaBackProjectionAlgorithm3D_DF::initializeFromProjector()
{
	m_iVoxelSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to BP3D_CUDA_DF");
		}
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaBackProjectionAlgorithm3D_DF::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaBackProjectionAlgorithm3D_DF", this, _cfg);	

	XMLNode node;
	int id;

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	//DF_X
	node = _cfg.self.getSingleNode("DeformXDataId");
	ASTRA_CONFIG_CHECK(node, "CudaForwardProjection3D", "No DeformXDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pDeformX = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("DeformXDataId");

	//DF_Y
	node = _cfg.self.getSingleNode("DeformYDataId");
	ASTRA_CONFIG_CHECK(node, "CudaForwardProjection3D", "No DeformYDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pDeformY = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("DeformYDataId");

	//DF_Z
	node = _cfg.self.getSingleNode("DeformZDataId");
	ASTRA_CONFIG_CHECK(node, "CudaForwardProjection3D", "No DeformZDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pDeformZ = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("DeformZDataId");

	initializeFromProjector();

	// Deprecated options
	m_iVoxelSuperSampling = (int)_cfg.self.getOptionNumerical("VoxelSuperSampling", m_iVoxelSuperSampling);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUindex", m_iGPUIndex);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("VoxelSuperSampling");
	CC.markOptionParsed("GPUIndex");
	if (!_cfg.self.hasOption("GPUIndex"))
		CC.markOptionParsed("GPUindex");



	m_bSIRTWeighting = _cfg.self.getOptionBool("SIRTWeighting", false);
	CC.markOptionParsed("SIRTWeighting");

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaBackProjectionAlgorithm3D_DF::initialize(CProjector3D* _pProjector, 
								  CFloat32ProjectionData3D* _pSinogram, 
								  CFloat32VolumeData3D* _pReconstruction)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	m_bSIRTWeighting = false;

	initializeFromProjector();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Iterate
void CCudaBackProjectionAlgorithm3D_DF::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	CFloat32ProjectionData3D* pSinoMem = dynamic_cast<CFloat32ProjectionData3D*>(m_pSinogram);
	ASTRA_ASSERT(pSinoMem);
	CFloat32VolumeData3D* pReconMem = dynamic_cast<CFloat32VolumeData3D*>(m_pReconstruction);
	ASTRA_ASSERT(pReconMem);
	CFloat32VolumeData3D* pDeformXMem = dynamic_cast<CFloat32VolumeData3D*>(m_pDeformX);
	ASTRA_ASSERT(pDeformXMem);
	CFloat32VolumeData3D* pDeformYMem = dynamic_cast<CFloat32VolumeData3D*>(m_pDeformY);
	ASTRA_ASSERT(pDeformYMem);
	CFloat32VolumeData3D* pDeformZMem = dynamic_cast<CFloat32VolumeData3D*>(m_pDeformZ);
	ASTRA_ASSERT(pDeformZMem);

	const CProjectionGeometry3D* projgeom = pSinoMem->getGeometry();
	const CVolumeGeometry3D& volgeom = *pReconMem->getGeometry();

	if (m_bSIRTWeighting) {
		CFloat32ProjectionData3DMemory* pSinoMemory = dynamic_cast<CFloat32ProjectionData3DMemory*>(m_pSinogram);
		ASTRA_ASSERT(pSinoMemory);
		CFloat32VolumeData3DMemory* pReconMemory = dynamic_cast<CFloat32VolumeData3DMemory*>(m_pReconstruction);
		ASTRA_ASSERT(pReconMemory);
		astraCudaBP_SIRTWeighted(pReconMemory->getData(),
		                         pSinoMemory->getDataConst(),
		                         &volgeom, projgeom,
		                         m_iGPUIndex, m_iVoxelSuperSampling);
	} else {

#if 1
		CCompositeGeometryManager cgm;

		cgm.doBP_DF(m_pProjector, pReconMem, pSinoMem, pDeformXMem, pDeformYMem, pDeformZMem);
#else
		astraCudaBP(pReconMem->getData(), pSinoMem->getDataConst(),
		            &volgeom, projgeom,
		            m_iGPUIndex, m_iVoxelSuperSampling);
#endif
	}

}


} // namespace astra
