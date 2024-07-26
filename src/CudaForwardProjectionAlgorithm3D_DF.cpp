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

#include "astra/CudaForwardProjectionAlgorithm3D_DF.h"

#ifdef ASTRA_CUDA

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
std::string CCudaForwardProjectionAlgorithm3D_DF::type = "FP3D_CUDA_DF";

//----------------------------------------------------------------------------------------
// Constructor
CCudaForwardProjectionAlgorithm3D_DF::CCudaForwardProjectionAlgorithm3D_DF() 
{
	m_bIsInitialized = false;
	m_iGPUIndex = -1;
	m_iDetectorSuperSampling = 1;
	m_pProjector = 0;
	m_pProjections = 0;
	m_pVolume = 0;
	m_pDeformX = 0;
	m_pDeformY = 0;
	m_pDeformZ = 0;

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaForwardProjectionAlgorithm3D_DF::~CCudaForwardProjectionAlgorithm3D_DF() 
{

}

//---------------------------------------------------------------------------------------
void CCudaForwardProjectionAlgorithm3D_DF::initializeFromProjector()
{
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to FP3D_CUDA");
		}
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaForwardProjectionAlgorithm3D_DF::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaForwardProjectionAlgorithm3D_DF", this, _cfg);	

	XMLNode node;
	int id;

	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "CudaForwardProjection3D", "No ProjectionDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pProjections = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("VolumeDataId");
	ASTRA_CONFIG_CHECK(node, "CudaForwardProjection3D", "No VolumeDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pVolume = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("VolumeDataId");

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

	// optional: projector
	node = _cfg.self.getSingleNode("ProjectorId");
	m_pProjector = 0;
	if (node) {
		id = StringUtil::stringToInt(node.getContent(), -1);
		m_pProjector = CProjector3DManager::getSingleton().get(id);
		if (!m_pProjector) {
			ASTRA_WARN("Optional parameter ProjectorId is not a valid id");
		}
	}
	CC.markNodeParsed("ProjectorId");

	initializeFromProjector();

	// Deprecated options
	try {
		m_iDetectorSuperSampling = _cfg.self.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaForwardProjection3D", "Supersampling options must be integers.");
	}
	CC.markOptionParsed("DetectorSuperSampling");
	// GPU number
	try {
		m_iGPUIndex = _cfg.self.getOptionInt("GPUindex", -1);
		m_iGPUIndex = _cfg.self.getOptionInt("GPUIndex", m_iGPUIndex);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaForwardProjection3D", "GPUIndex must be an integer.");
	}
	CC.markOptionParsed("GPUIndex");
	if (!_cfg.self.hasOption("GPUIndex"))
		CC.markOptionParsed("GPUindex");

	// success
	m_bIsInitialized = check();

	if (!m_bIsInitialized)
		return false;

	return true;	
}


bool CCudaForwardProjectionAlgorithm3D_DF::initialize(CProjector3D* _pProjector, 
                                  CFloat32ProjectionData3D* _pProjections, 
                                  CFloat32VolumeData3D* _pVolume,
                                  int _iGPUindex, int _iDetectorSuperSampling)
{
	m_pProjector = _pProjector;
	
	// required classes
	m_pProjections = _pProjections;
	m_pVolume = _pVolume;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		// TODO: Report
		m_iDetectorSuperSampling = _iDetectorSuperSampling;
		m_iGPUIndex = _iGPUindex;
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

	// success
	m_bIsInitialized = check();

	if (!m_bIsInitialized)
		return false;

	return true;
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaForwardProjectionAlgorithm3D_DF::check() 
{
	// check pointers
	//ASTRA_CONFIG_CHECK(m_pProjector, "Reconstruction2D", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pProjections, "FP3D_CUDA_DF", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pVolume, "FP3D_CUDA_DF", "Invalid Volume Data Object.");
	ASTRA_CONFIG_CHECK(m_pDeformX, "FP3D_CUDA_DF", "Invalid DeformX Data Object.");
	ASTRA_CONFIG_CHECK(m_pDeformY, "FP3D_CUDA_DF", "Invalid DeformY Data Object.");
	ASTRA_CONFIG_CHECK(m_pDeformZ, "FP3D_CUDA_DF", "Invalid DeformZ Data Object.");


	// check initializations
	//ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "Reconstruction2D", "Projector Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pProjections->isInitialized(), "FP3D_CUDA_DF", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pVolume->isInitialized(), "FP3D_CUDA_DF", "Volume Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pDeformX->isInitialized(), "FP3D_CUDA_DF", "DeformX Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pDeformY->isInitialized(), "FP3D_CUDA_DF", "DeformY Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pDeformZ->isInitialized(), "FP3D_CUDA_DF", "DeformZ Data Object Not Initialized.");

	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "FP3D_CUDA_DF", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FP3D_CUDA_DF", "GPUIndex must be a non-negative integer.");

	// check compatibility between projector and data classes
//	ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "SIRT_CUDA", "Projection Data not compatible with the specified Projector.");
//	ASTRA_CONFIG_CHECK(m_pReconstruction->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "SIRT_CUDA", "Reconstruction Data not compatible with the specified Projector.");

	// todo: turn some of these back on

// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "SIRT_CUDA", "ProjectionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "SIRT_CUDA", "ProjectionGeometry not initialized.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry, "SIRT_CUDA", "ReconstructionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry->isInitialized(), "SIRT_CUDA", "ReconstructionGeometry not initialized.");

	// check dimensions
	//ASTRA_CONFIG_CHECK(m_pSinogram->getAngleCount() == m_pProjectionGeometry->getProjectionAngleCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pSinogram->getDetectorCount() == m_pProjectionGeometry->getDetectorCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getWidth() == m_pReconstructionGeometry->getGridColCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getHeight() == m_pReconstructionGeometry->getGridRowCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	
	// check restrictions
	// TODO: check restrictions built into cuda code

	// success
	m_bIsInitialized = true;
	return true;
}


void CCudaForwardProjectionAlgorithm3D_DF::setGPUIndex(int _iGPUIndex)
{
	m_iGPUIndex = _iGPUIndex;
}

//----------------------------------------------------------------------------------------
// Run
void CCudaForwardProjectionAlgorithm3D_DF::run(int)
{
	// check initialized
	assert(m_bIsInitialized);

#if 1
	CCompositeGeometryManager cgm;

	cgm.doFP_DF(m_pProjector, m_pVolume, m_pProjections,m_pDeformX,m_pDeformY,m_pDeformZ);

#else
	const CProjectionGeometry3D* projgeom = m_pProjections->getGeometry();
	const CVolumeGeometry3D& volgeom = *m_pVolume->getGeometry();

	Cuda3DProjectionKernel projKernel = ker3d_default;
	if (m_pProjector) {
		CCudaProjector3D* projector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
		projKernel = projector->getProjectionKernel();
	}

#if 0
	// Debugging code that gives the coordinates of the corners of the volume
	// projected on the detector.
	{
		float fX[] = { volgeom.getWindowMinX(), volgeom.getWindowMaxX() };
		float fY[] = { volgeom.getWindowMinY(), volgeom.getWindowMaxY() };
		float fZ[] = { volgeom.getWindowMinZ(), volgeom.getWindowMaxZ() };

		for (int a = 0; a < projgeom->getProjectionCount(); ++a)
		for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
		for (int k = 0; k < 2; ++k) {
			float fU, fV;
			projgeom->projectPoint(fX[i], fY[j], fZ[k], a, fU, fV);
			ASTRA_DEBUG("%3d %c1,%c1,%c1 -> %12f %12f", a, i ? ' ' : '-', j ? ' ' : '-', k ? ' ' : '-', fU, fV);
		}
	}
#endif

	astraCudaFP(m_pVolume->getDataConst(), m_pProjections->getData(),
	            &volgeom, projgeom,
	            m_iGPUIndex, m_iDetectorSuperSampling, projKernel);
#endif
}


}

#endif
