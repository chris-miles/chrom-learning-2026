load(['rpe18_prc_736.mat']);
CS=centrioles; CS1x=0.5*(CS(:,1,1)+CS(:,4,1));CS1y=0.5*(CS(:,2,1)+CS(:,5,1));
CS2x=0.5*(CS(:,1,2)+CS(:,4,2));CS2y=0.5*(CS(:,2,2)+CS(:,5,2));
KT=kinetochores;KTx=0*KT(:,1,1); for j=1:46 KTx=KTx+0.5*(KT(:,1,j)+KT(:,4,j)); end, 
KTx=KTx/46;
KTy=0*KT(:,1,1); for j=1:46 KTy=KTy+0.5*(KT(:,2,j)+KT(:,5,j)); end, KTy=KTy/46;
CSx=0.5*(CS1x+CS2x);CSy=0.5*(CS1y+CS2y);
%plot(CSx,CSy,'.r',KTx,KTy,'.b',CSx(1),CSy(1),'*g',KTx(1),KTy(1),'*y')

dCSx=diff(CSx);dCSy=diff(CSy); P=zeros(100,1);
dKTx=diff(KTx);dKTy=diff(KTy); C=zeros(100,1);
for s=1:100 v1=[dCSx(s) dCSy(s)]; z=[(KTx(s)-CSx(s)) (KTy(s)-CSy(s))];
    v2=[dKTx(s) dKTy(s)]; P(s)=dot(v1,z)/(norm(z)); %norm(v1)*
    C(s)=dot(v2,-z)/(norm(z)); end %norm(v1)* 

s=100;subplot(3,1,1), plot(CSx(1:s),CSy(1:s),'.r',KTx(1:s),KTy(1:s),'.b'),axis equal
subplot(3,1,2), hist(P)
subplot(3,1,3), hist(C)
PP=[PP;P];